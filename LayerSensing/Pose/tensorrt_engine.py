"""TensorRT-backed YOLOv8 pose inference utilities."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch

from ultralytics.yolo.utils import ops

LOGGER = logging.getLogger(__name__)


@dataclass
class PoseDetection:
    """Single pose detection result."""

    bbox: List[float]
    score: float
    class_id: int
    keypoints: List[List[float]]


@dataclass
class PoseInferenceResult:
    """Container for pose detections and runtime profiling information."""

    detections: List[PoseDetection]
    timings_ms: Dict[str, float]


class TensorRTPoseEngine:
    """Runs YOLOv8 pose models exported as TensorRT engines."""

    def __init__(
        self,
        engine_path: Path | str,
        *,
        input_shape: Tuple[int, int, int] = (3, 640, 640),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_det: int = 100,
        num_keypoints: int = 17,
        num_classes: int = 1,
        warmup: bool = True,
    ) -> None:
        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self.input_shape = input_shape
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes

        self._trt, self._cudart = self._import_runtime()
        self._logger = self._create_logger()
        self._runtime = self._trt.Runtime(self._logger)
        with self.engine_path.open("rb") as engine_file:
            engine_bytes = engine_file.read()
        self._log_engine_context(engine_bytes)
        self._engine = self._deserialize_engine(engine_bytes)

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        _, self._stream = self._cudart.cudaStreamCreate()
        self._device_mem: Dict[int, int] = {}
        self._host_mem: Dict[int, np.ndarray] = {}
        self._host_shape: Dict[int, Tuple[int, ...]] = {}

        # TensorRT 10.x removes the legacy binding APIs. Build a compatibility
        # layer that works for both the legacy (get_binding_index) and the new
        # tensor APIs (get_tensor_index / get_tensor_mode).
        self._binding_names = list(self._engine)
        self._binding_indices = {name: idx for idx, name in enumerate(self._binding_names)}

        if hasattr(self._engine, "get_binding_index"):
            self._index_of = self._engine.get_binding_index  # type: ignore[attr-defined]
            self._is_input = self._engine.binding_is_input  # type: ignore[attr-defined]
            self._dtype_of = self._engine.get_binding_dtype  # type: ignore[attr-defined]
            self._shape_of = self._engine.get_binding_shape  # type: ignore[attr-defined]
            self._num_bindings = getattr(self._engine, "num_bindings", len(self._binding_names))
            use_tensor_api = False
        elif hasattr(self._engine, "get_tensor_index"):
            trt = self._trt

            def _index_of(name: str) -> int:
                return self._engine.get_tensor_index(name)  # type: ignore[attr-defined]

            def _is_input(name: str) -> bool:
                mode = self._engine.get_tensor_mode(name)  # type: ignore[attr-defined]
                return mode == trt.TensorIOMode.INPUT

            def _dtype_of(name: str):
                return self._engine.get_tensor_dtype(name)  # type: ignore[attr-defined]

            def _shape_of(name: str):
                return self._engine.get_tensor_shape(name)  # type: ignore[attr-defined]

            self._index_of = _index_of
            self._is_input = _is_input
            self._dtype_of = _dtype_of
            self._shape_of = _shape_of
            self._num_bindings = getattr(self._engine, "num_io_tensors", len(self._binding_names))
            use_tensor_api = True
        else:
            raise RuntimeError(
                "TensorRT engine does not expose binding/tensor introspection APIs "
                "(expected get_binding_index or get_tensor_index)."
            )

        self._input_names = [name for name in self._binding_names if self._is_input(name)]
        if not self._input_names:
            raise RuntimeError("Pose engine exposes no input tensors")
        self._input_name = self._input_names[0]
        self._input_index = self._index_of(self._input_name)

        self._output_names = [name for name in self._binding_names if not self._is_input(name)]
        if len(self._output_names) != 1:
            raise RuntimeError(
                "Pose engine is expected to expose exactly one output tensor; "
                f"got {len(self._output_names)} bindings."
            )
        self._output_name = self._output_names[0]
        self._output_index = self._index_of(self._output_name)

        self._input_dtype = np.dtype(self._trt.nptype(self._dtype_of(self._input_name)))
        self._output_dtype = np.dtype(self._trt.nptype(self._dtype_of(self._output_name)))

        self._use_tensor_api = use_tensor_api

        if warmup:
            LOGGER.debug("Running TensorRT pose warmup inference")
            dummy = np.zeros((1, *self.input_shape), dtype=self._input_dtype)
            self._run_inference(dummy)

    # ---------------------------------------------------------------------
    @staticmethod
    def _import_runtime():
        try:
            import tensorrt as trt  # type: ignore
        except OSError as exc:  # pragma: no cover - hardware dependency
            raise RuntimeError(
                "TensorRT runtime libraries could not be loaded. The CUDA/cuDNN shared libraries (e.g. "
                "libcublas.so.11, libcudnn_ops_infer.so.8) must be installed and compatible with the "
                "TensorRT version (10.7.0). Install the matching CUDA runtime/cuDNN packages on the host "
                "or add NVIDIA's runtime wheels (e.g. `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`) "
                "before launching the pose worker."
            ) from exc
        except ImportError as exc:  # pragma: no cover - hardware dependency
            raise RuntimeError(
                "TensorRT Python bindings are required to use the quantized pose engine. Install them with "
                "`pip install tensorrt==10.7.0` (and ensure compatible CUDA/cuDNN runtime libraries are "
                "present)."
            ) from exc

        try:
            from cuda import cudart  # type: ignore
        except ImportError as exc:  # pragma: no cover - hardware dependency
            raise RuntimeError(
                "cuda-python is required for TensorRT execution. Install it with `pip install cuda-python`."
            ) from exc

        return trt, cudart

    # ------------------------------------------------------------------
    def _create_logger(self):
        trt = self._trt

        class CaptureLogger(trt.ILogger):  # type: ignore[name-defined]
            def __init__(self):
                super().__init__()
                self.errors = []

            def log(self, severity, msg):  # pragma: no cover - hardware dependency
                text = str(msg)
                if severity <= self.Severity.WARNING:
                    LOGGER.warning("TensorRT: %s", text)
                if severity <= self.Severity.ERROR:
                    self.errors.append(text)

        return CaptureLogger()

    # ------------------------------------------------------------------
    def _deserialize_engine(self, engine_bytes: bytes):
        if not self._validate_engine_bytes(engine_bytes):
            raise RuntimeError(self._format_engine_error(None))
        try:
            engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        except Exception as exc:  # pragma: no cover - hardware dependency
            raise RuntimeError(self._format_engine_error(exc)) from exc

        if engine is None:  # pragma: no cover - hardware dependency
            raise RuntimeError(self._format_engine_error(None))

        return engine

    # ------------------------------------------------------------------
    def _format_engine_error(self, exc: Exception | None) -> str:
        trt_error = "; ".join(self._logger.errors) if getattr(self._logger, "errors", None) else "Unknown TensorRT error"
        hint = (
            "TensorRT could not deserialize the engine. This usually means the file is "
            "corrupted or was built with a different TensorRT version. Rebuild the engine "
            f"with the same TensorRT version as the runtime ({self._trt.__version__})."
        )
        base = f"Failed to load TensorRT engine: {self.engine_path}. TensorRT error: {trt_error}. {hint}"
        if exc is not None:
            return f"{base} Original exception: {exc}"
        return base

    # ------------------------------------------------------------------
    def _validate_engine_bytes(self, engine_bytes: bytes) -> bool:
        if len(engine_bytes) < 1024:  # pragma: no cover - defensive check
            self._logger.errors.append(
                f"Engine file is unexpectedly small ({len(engine_bytes)} bytes); it may be incomplete or corrupted."
            )
            return False
        return True

    # ------------------------------------------------------------------
    def _log_engine_context(self, engine_bytes: bytes) -> None:
        try:
            stats = self.engine_path.stat()
            size_mb = stats.st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(stats.st_mtime).isoformat(timespec="seconds")
            LOGGER.info(
                "Loading TensorRT pose engine %s (%.2f MB, mtime=%s) with runtime %s",
                self.engine_path,
                size_mb,
                mod_time,
                self._trt.__version__,
            )
        except Exception:  # pragma: no cover - best-effort logging
            pass
        if len(engine_bytes) < 1024:  # pragma: no cover - defensive trace
            LOGGER.warning("TensorRT engine file is only %d bytes; deserialization will likely fail", len(engine_bytes))

    # ------------------------------------------------------------------
    def close(self) -> None:
        for ptr in self._device_mem.values():  # pragma: no cover - hardware dependency
            self._cudart.cudaFree(ptr)
        self._device_mem.clear()
        self._host_mem.clear()
        self._host_shape.clear()
        if getattr(self, "_stream", None) is not None:
            self._cudart.cudaStreamDestroy(self._stream)
            self._stream = None
        self._context = None
        self._engine = None
        self._runtime = None

    # ------------------------------------------------------------------
    def __del__(self):  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:  # pylint: disable=broad-except
            pass

    # ------------------------------------------------------------------
    def predict(self, image: np.ndarray) -> PoseInferenceResult:
        """Run pose estimation on a single image."""
        prep_start = time.perf_counter()
        input_tensor, meta = self._preprocess(image)
        preprocess_ms = (time.perf_counter() - prep_start) * 1000.0

        infer_start = time.perf_counter()
        raw_output = self._run_inference(input_tensor)
        inference_ms = (time.perf_counter() - infer_start) * 1000.0

        post_start = time.perf_counter()
        detections = self._postprocess(raw_output, meta)
        postprocess_ms = (time.perf_counter() - post_start) * 1000.0

        timings = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
        }
        return PoseInferenceResult(detections=detections, timings_ms=timings)

    # ------------------------------------------------------------------
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Pose engine expects color images in HxWx3 format")

        h, w = image.shape[:2]
        target_h, target_w = self.input_shape[1:]
        if h == 0 or w == 0:
            raise ValueError("Empty image provided to pose engine")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized, scale, (pad_x, pad_y) = self._letterbox(image_rgb, (target_h, target_w))
        tensor = resized.astype(self._input_dtype)
        if self._input_dtype in (np.float16, np.float32):
            tensor = tensor / np.array(255.0, dtype=self._input_dtype)
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])
        return tensor, {
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "orig_h": float(h),
            "orig_w": float(w),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _letterbox(
        image: np.ndarray,
        new_shape: Tuple[int, int],
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        shape = image.shape[:2]  # h, w
        if not shape[0] or not shape[1]:
            raise ValueError("Invalid image shape for letterbox")

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded, r, (left, top)

    # ------------------------------------------------------------------
    def _ensure_allocation(self, index: int, shape: Iterable[int], dtype: np.dtype) -> None:
        shape_tuple = tuple(int(dim) for dim in shape)
        size = int(np.prod(shape_tuple))
        if index in self._host_mem and self._host_mem[index].size == size:
            return

        if index in self._device_mem:
            self._cudart.cudaFree(self._device_mem[index])

        nbytes = size * dtype.itemsize
        _, device_ptr = self._cudart.cudaMalloc(nbytes)
        self._device_mem[index] = device_ptr
        self._host_mem[index] = np.empty(size, dtype=dtype)
        self._host_shape[index] = shape_tuple

    # ------------------------------------------------------------------
    def _run_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        batch_shape = tuple(input_tensor.shape)
        if self._use_tensor_api and hasattr(self._context, "set_input_shape"):
            self._context.set_input_shape(self._input_name, batch_shape)
        else:
            self._context.set_binding_shape(self._input_index, batch_shape)
        self._ensure_allocation(self._input_index, batch_shape, self._input_dtype)

        np.copyto(self._host_mem[self._input_index].reshape(batch_shape), input_tensor)
        self._cudart.cudaMemcpy(
            self._device_mem[self._input_index],
            self._host_mem[self._input_index].ctypes.data,
            self._host_mem[self._input_index].nbytes,
            self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

        if self._use_tensor_api and hasattr(self._context, "get_tensor_shape"):
            output_shape = self._context.get_tensor_shape(self._output_name)
        else:
            output_shape = self._context.get_binding_shape(self._output_index)
        if not output_shape:
            output_shape = self._shape_of(self._output_name)
        self._ensure_allocation(self._output_index, output_shape, self._output_dtype)

        bindings = [0] * self._num_bindings
        for name in self._binding_names:
            idx = self._index_of(name)
            bindings[idx] = self._device_mem[idx]

        self._context.execute_async_v2(bindings=bindings, stream_handle=self._stream)
        self._cudart.cudaMemcpy(
            self._host_mem[self._output_index].ctypes.data,
            self._device_mem[self._output_index],
            self._host_mem[self._output_index].nbytes,
            self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        self._cudart.cudaStreamSynchronize(self._stream)

        return self._host_mem[self._output_index].reshape(self._host_shape[self._output_index])

    # ------------------------------------------------------------------
    def _postprocess(self, raw_output: np.ndarray, meta: Dict[str, float]) -> List[PoseDetection]:
        if raw_output.ndim == 2:
            pred = torch.from_numpy(raw_output[None, ...])
        elif raw_output.ndim == 3:
            pred = torch.from_numpy(raw_output)
        else:
            raise ValueError(f"Unexpected output shape from TensorRT engine: {raw_output.shape}")

        if pred.shape[1] < pred.shape[2]:
            pred = pred.permute(0, 2, 1)
        pred = pred.float()

        preds = ops.non_max_suppression(
            pred,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            max_det=self.max_det,
            nc=self.num_classes,
            nkpt=self.num_keypoints,
            kpt_label=True,
            multi_label=False,
            agnostic=False,
        )

        detections: List[PoseDetection] = []
        det = preds[0]
        if det is None or not len(det):
            return detections

        gain = meta["scale"]
        pad_x = meta["pad_x"]
        pad_y = meta["pad_y"]
        orig_h = meta["orig_h"]
        orig_w = meta["orig_w"]

        det = det.cpu().numpy()
        boxes = det[:, :4]
        scores = det[:, 4]
        classes = det[:, 5].astype(int, copy=False)
        kpt_array = det[:, 6:].reshape(-1, self.num_keypoints, 3)

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / gain
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / gain
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h - 1)

        kpt_array[..., 0] = (kpt_array[..., 0] - pad_x) / gain
        kpt_array[..., 1] = (kpt_array[..., 1] - pad_y) / gain
        kpt_array[..., 0] = kpt_array[..., 0].clip(0, orig_w - 1)
        kpt_array[..., 1] = kpt_array[..., 1].clip(0, orig_h - 1)

        for box, score, class_id, keypoints in zip(boxes, scores, classes, kpt_array):
            detections.append(
                PoseDetection(
                    bbox=[float(x) for x in box.tolist()],
                    score=float(score),
                    class_id=int(class_id),
                    keypoints=[[float(v) for v in kp.tolist()] for kp in keypoints],
                )
            )
        return detections


__all__ = ["TensorRTPoseEngine", "PoseDetection", "PoseInferenceResult"]
