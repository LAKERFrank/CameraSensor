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
        self._load_ok = True
        logging.info('Pose engine loaded: %s', self.engine_path)
        return True

    def infer(self, image: np.ndarray) -> List[PoseDetection]:
        if not self._load_ok:
            return []
        input_tensor, ratio, pad = self._preprocess(image)
        # TODO: real TensorRT bindings execution.
        # Keep pipeline stable even when engine I/O mapping is not finalized.
        raw = np.empty((0, 57), dtype=np.float32)
        return self._postprocess(raw, image.shape[:2], ratio, pad)

    def _preprocess(self, image):
        h, w = image.shape[:2]
        target_w, target_h = self.input_size
        ratio = min(target_w / w, target_h / h)
        nw, nh = int(round(w * ratio)), int(round(h * ratio))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        dw = (target_w - nw) // 2
        dh = (target_h - nh) // 2
        canvas[dh:dh + nh, dw:dw + nw, :] = resized
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
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
