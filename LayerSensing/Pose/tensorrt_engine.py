"""TensorRT-backed YOLOv8 pose inference utilities."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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
        self._runtime = self._trt.Runtime(self._trt.Logger(self._trt.Logger.WARNING))
        with self.engine_path.open("rb") as engine_file:
            self._engine = self._runtime.deserialize_cuda_engine(engine_file.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.engine_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        _, self._stream = self._cudart.cudaStreamCreate()
        self._device_mem: Dict[int, int] = {}
        self._host_mem: Dict[int, np.ndarray] = {}
        self._host_shape: Dict[int, Tuple[int, ...]] = {}

        self._input_index = self._engine.get_binding_index(self._engine[0])
        self._output_indices = [
            self._engine.get_binding_index(name)
            for name in self._engine if not self._engine.binding_is_input(name)
        ]
        if len(self._output_indices) != 1:
            raise RuntimeError(
                "Pose engine is expected to expose exactly one output tensor; "
                f"got {len(self._output_indices)} bindings."
            )
        self._output_index = self._output_indices[0]

        self._input_dtype = np.dtype(self._trt.nptype(self._engine.get_binding_dtype(self._input_index)))
        self._output_dtype = np.dtype(self._trt.nptype(self._engine.get_binding_dtype(self._output_index)))

        # Cache calibration ranges so that INT8 engines can be quantized/dequantized properly.
        self._input_dynamic_range = self._engine.get_dynamic_range(self._input_index)
        self._output_dynamic_range = self._engine.get_dynamic_range(self._output_index)
        self._normalize_inputs = True
        self._center_inputs = False
        self._input_quant_scale = None
        if self._input_dtype == np.int8:
            if self._input_dynamic_range is None:
                raise RuntimeError(
                    "Quantized pose engine is missing calibration ranges for the input binding."
                )
            input_min, input_max = self._input_dynamic_range
            if input_min is None or input_max is None:
                raise RuntimeError(
                    "Quantized pose engine returned invalid calibration ranges for the input binding."
                )
            # Determine whether the original preprocessing normalized to [0, 1].
            # Most YOLOv8 exports do, in which case the calibration max will be close to 1.
            self._normalize_inputs = input_max <= 2.0
            # When calibration uses symmetric ranges (e.g. [-1, 1]) we must center the inputs.
            self._center_inputs = input_min < 0.0
            # Convert calibration range to a quantization scale.
            denom = max(abs(input_min), abs(input_max))
            if denom == 0:
                raise RuntimeError("Quantized pose engine reported zero dynamic range for the input binding.")
            self._input_quant_scale = 127.0 / denom

        if warmup:
            LOGGER.debug("Running TensorRT pose warmup inference")
            dummy = np.zeros((1, *self.input_shape), dtype=self._input_dtype)
            self._run_inference(dummy)

    # ---------------------------------------------------------------------
    @staticmethod
    def _import_runtime():
        try:
            import tensorrt as trt  # type: ignore
        except ImportError as exc:  # pragma: no cover - hardware dependency
            raise RuntimeError(
                "TensorRT Python bindings are required to use the quantized pose engine. "
                "Install them via `pip install --index-url https://pypi.ngc.nvidia.com nvidia-tensorrt`."
            ) from exc

        try:
            from cuda import cudart  # type: ignore
        except ImportError as exc:  # pragma: no cover - hardware dependency
            raise RuntimeError(
                "cuda-python is required for TensorRT execution. Install it with `pip install cuda-python`."
            ) from exc

        return trt, cudart

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
        tensor = resized.astype(np.float32)
        if self._normalize_inputs:
            tensor /= 255.0
        if self._center_inputs:
            tensor = tensor * 2.0 - 1.0
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])
        tensor = self._convert_input_dtype(tensor)
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
        self._context.set_binding_shape(self._input_index, batch_shape)
        self._ensure_allocation(self._input_index, batch_shape, self._input_dtype)

        np.copyto(self._host_mem[self._input_index].reshape(batch_shape), input_tensor)
        self._cudart.cudaMemcpy(
            self._device_mem[self._input_index],
            self._host_mem[self._input_index].ctypes.data,
            self._host_mem[self._input_index].nbytes,
            self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

        output_shape = self._context.get_binding_shape(self._output_index)
        if not output_shape:
            output_shape = self._engine.get_binding_shape(self._output_index)
        self._ensure_allocation(self._output_index, output_shape, self._output_dtype)

        bindings = [0] * self._engine.num_bindings
        for name in self._engine:
            idx = self._engine.get_binding_index(name)
            bindings[idx] = self._device_mem[idx]

        self._context.execute_async_v2(bindings=bindings, stream_handle=self._stream)
        self._cudart.cudaMemcpy(
            self._host_mem[self._output_index].ctypes.data,
            self._device_mem[self._output_index],
            self._host_mem[self._output_index].nbytes,
            self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        self._cudart.cudaStreamSynchronize(self._stream)

        output = self._host_mem[self._output_index].reshape(self._host_shape[self._output_index])
        return self._convert_output_dtype(output)

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

    # ------------------------------------------------------------------
    def _convert_input_dtype(self, tensor: np.ndarray) -> np.ndarray:
        if self._input_dtype in (np.float32, np.float16):
            return tensor.astype(self._input_dtype)

        if self._input_dtype == np.int8:
            if self._input_quant_scale is None:
                raise RuntimeError("Quantized pose engine was not initialised correctly")
            quantized = np.clip(np.round(tensor * self._input_quant_scale), -128, 127)
            return quantized.astype(np.int8)

        raise NotImplementedError(
            f"Unsupported pose engine input dtype: {self._input_dtype!r}."
        )

    # ------------------------------------------------------------------
    def _convert_output_dtype(self, tensor: np.ndarray) -> np.ndarray:
        if tensor.dtype == np.float32:
            return tensor
        if tensor.dtype == np.float16:
            return tensor.astype(np.float32)
        if tensor.dtype == np.int8:
            if self._output_dynamic_range is None:
                raise RuntimeError(
                    "Quantized pose engine is missing calibration ranges for the output binding."
                )
            output_min, output_max = self._output_dynamic_range
            if output_min is None or output_max is None:
                raise RuntimeError(
                    "Quantized pose engine returned invalid calibration ranges for the output binding."
                )
            denom = max(abs(output_min), abs(output_max))
            if denom == 0:
                raise RuntimeError("Quantized pose engine reported zero dynamic range for the output binding.")
            scale = denom / 127.0
            return tensor.astype(np.float32) * scale

        raise NotImplementedError(
            f"Unsupported pose engine output dtype: {tensor.dtype!r}."
        )


__all__ = ["TensorRTPoseEngine", "PoseDetection", "PoseInferenceResult"]
