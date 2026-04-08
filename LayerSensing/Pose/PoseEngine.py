import logging
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class PoseDetection:
    bbox_xywh: list
    bbox_conf: float
    keypoints: list


class TensorRTPoseEngine:
    def __init__(self, engine_path: str, input_size=(640, 640), conf_thres: float = 0.25, kpt_thres: float = 0.1):
        self.engine_path = engine_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.kpt_thres = kpt_thres
        self.runtime = None
        self.engine = None
        self.context = None
        self.bindings = None
        self.stream = None
        self._load_ok = False
        self.trt = None
        self.cuda = None
        self.input_name = None
        self.output_names = []

    def load(self):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except Exception as e:
            logging.error('TensorRT dependencies not available: %s', e)
            return False

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            logging.error('Failed to deserialize pose engine: %s', self.engine_path)
            return False

        context = engine.create_execution_context()
        if context is None:
            logging.error('Failed to create execution context for pose engine')
            return False

        self.runtime = runtime
        self.engine = engine
        self.context = context
        self.trt = trt
        self.cuda = cuda
        self.stream = cuda.Stream()
        self.input_name, self.output_names = self._discover_io_names()
        if self.input_name is None or len(self.output_names) == 0:
            logging.error('Failed to discover TensorRT Pose I/O tensors')
            return False
        self._load_ok = True
        logging.info('Pose engine loaded: %s', self.engine_path)
        return True

    def infer(self, image: np.ndarray) -> List[PoseDetection]:
        if not self._load_ok:
            return []
        input_tensor, ratio, pad = self._preprocess(image)
        raw = self._execute_trt(input_tensor)
        return self._postprocess(raw, image.shape[:2], ratio, pad)

    def _discover_io_names(self):
        input_name = None
        output_names = []

        if hasattr(self.engine, 'num_io_tensors'):
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                mode = self.engine.get_tensor_mode(name)
                if mode == self.trt.TensorIOMode.INPUT:
                    input_name = name
                elif mode == self.trt.TensorIOMode.OUTPUT:
                    output_names.append(name)
        else:
            for idx in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(idx)
                if self.engine.binding_is_input(idx):
                    input_name = name
                else:
                    output_names.append(name)

        return input_name, output_names

    def _execute_trt(self, input_tensor: np.ndarray) -> np.ndarray:
        assert self.context is not None and self.engine is not None

        if hasattr(self.context, 'set_input_shape'):
            self.context.set_input_shape(self.input_name, tuple(input_tensor.shape))
        else:
            binding_idx = self.engine.get_binding_index(self.input_name)
            self.context.set_binding_shape(binding_idx, tuple(input_tensor.shape))

        io = self._prepare_io_buffers(input_tensor)

        if hasattr(self.context, 'set_tensor_address'):
            for item in io:
                self.context.set_tensor_address(item['name'], int(item['device']))
            success = self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            bindings = [0] * self.engine.num_bindings
            for item in io:
                bindings[item['index']] = int(item['device'])
            success = self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        if not success:
            raise RuntimeError('TensorRT execute failed for pose engine')

        for item in io:
            if not item['is_input']:
                self.cuda.memcpy_dtoh_async(item['host'], item['device'], self.stream)
        self.stream.synchronize()

        output_arrays = [item['host'].reshape(item['shape']) for item in io if not item['is_input']]
        if len(output_arrays) == 0:
            return np.empty((0, 57), dtype=np.float32)
        return self._decode_raw_output(output_arrays[0])

    def _prepare_io_buffers(self, input_tensor: np.ndarray):
        io = []

        if hasattr(self.engine, 'num_io_tensors'):
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                mode = self.engine.get_tensor_mode(name)
                is_input = mode == self.trt.TensorIOMode.INPUT
                dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dim < 0 for dim in shape):
                    raise RuntimeError(f'Unresolved dynamic shape for tensor {name}: {shape}')
                size = int(np.prod(shape))
                host = self.cuda.pagelocked_empty(size, dtype)
                device = self.cuda.mem_alloc(host.nbytes)
                if is_input:
                    np.copyto(host, input_tensor.astype(dtype, copy=False).reshape(-1))
                    self.cuda.memcpy_htod_async(device, host, self.stream)
                io.append({'name': name, 'index': idx, 'is_input': is_input, 'shape': shape, 'host': host, 'device': device})
        else:
            for idx in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(idx)
                is_input = self.engine.binding_is_input(idx)
                dtype = self.trt.nptype(self.engine.get_binding_dtype(idx))
                shape = tuple(self.context.get_binding_shape(idx))
                if any(dim < 0 for dim in shape):
                    raise RuntimeError(f'Unresolved dynamic shape for binding {name}: {shape}')
                size = int(np.prod(shape))
                host = self.cuda.pagelocked_empty(size, dtype)
                device = self.cuda.mem_alloc(host.nbytes)
                if is_input:
                    np.copyto(host, input_tensor.astype(dtype, copy=False).reshape(-1))
                    self.cuda.memcpy_htod_async(device, host, self.stream)
                io.append({'name': name, 'index': idx, 'is_input': is_input, 'shape': shape, 'host': host, 'device': device})

        return io

    def _decode_raw_output(self, output: np.ndarray) -> np.ndarray:
        raw = np.asarray(output, dtype=np.float32)
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        # common layout from YOLO-based TRT export: [C, N]
        if raw.ndim == 2 and raw.shape[0] in (56, 57) and raw.shape[1] > raw.shape[0]:
            raw = raw.T

        # convert 56-col format [cx,cy,w,h,conf,kpts(51)] to existing 57-col parser format
        if raw.ndim == 2 and raw.shape[1] == 56:
            zeros = np.zeros((raw.shape[0], 1), dtype=raw.dtype)
            raw = np.concatenate([raw[:, :5], zeros, raw[:, 5:]], axis=1)

        if raw.ndim != 2 or raw.shape[1] < 57:
            logging.warning('Unexpected pose output shape: %s, fallback to empty detections', raw.shape)
            return np.empty((0, 57), dtype=np.float32)

        return raw[:, :57]

    def _preprocess(self, image):
        if image.ndim == 2:
            channels = 1
        elif image.ndim == 3 and image.shape[2] > 0:
            channels = image.shape[2]
        else:
            raise ValueError(f'Unsupported image shape for pose preprocess: {image.shape}')

        h, w = image.shape[:2]
        target_w, target_h = self.input_size
        ratio = min(target_w / w, target_h / h)
        nw, nh = int(round(w * ratio)), int(round(h * ratio))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw = (target_w - nw) // 2
        dh = (target_h - nh) // 2

        if channels == 1:
            if resized.ndim == 3:
                resized = resized[:, :, 0]
            canvas = np.full((target_h, target_w), 114, dtype=np.uint8)
            canvas[dh:dh + nh, dw:dw + nw] = resized
            tensor = canvas[np.newaxis, :, :].astype(np.float32) / 255.0
        else:
            canvas = np.full((target_h, target_w, channels), 114, dtype=np.uint8)
            canvas[dh:dh + nh, dw:dw + nw, :] = resized
            if channels == 3:
                tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            else:
                tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0

        return np.expand_dims(tensor, 0), ratio, (dw, dh)

    def _postprocess(self, raw, orig_shape, ratio, pad):
        oh, ow = orig_shape
        detections = []
        for row in raw:
            conf = float(row[4])
            if conf < self.conf_thres:
                continue
            cx, cy, w, h = row[:4]
            cx = (cx - pad[0]) / ratio
            cy = (cy - pad[1]) / ratio
            w = w / ratio
            h = h / ratio
            keypoints = []
            kpt = row[6:].reshape(-1, 3)[:17]
            for x, y, kconf in kpt:
                x = (x - pad[0]) / ratio
                y = (y - pad[1]) / ratio
                if kconf < self.kpt_thres:
                    keypoints.append([0.0, 0.0, 0.0])
                else:
                    keypoints.append([float(np.clip(x, 0, ow - 1)), float(np.clip(y, 0, oh - 1)), float(kconf)])
            detections.append(PoseDetection(
                bbox_xywh=[float(np.clip(cx, 0, ow - 1)), float(np.clip(cy, 0, oh - 1)), float(max(w, 0.0)), float(max(h, 0.0))],
                bbox_conf=conf,
                keypoints=keypoints,
            ))
        detections.sort(key=lambda d: d.bbox_conf, reverse=True)
        return detections
