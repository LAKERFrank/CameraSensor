import json
import logging
import os
import threading
import time

import cv2
import numpy as np

from LayerSensing.Pose.PoseEngine import TensorRTPoseEngine


class PoseMqtt(threading.Thread):
    def __init__(self, nodename, data_handler, frame_queue, engine_path: str, vis_dir: str = "", cam_idx: int = 0):
        super().__init__(name=nodename)
        self.nodename = nodename
        self.data_handler = data_handler
        self.frame_queue = frame_queue
        self.engine = TensorRTPoseEngine(engine_path=engine_path)
        self._stopper = threading.Event()
        self._counter = 0
        self._lat_acc = 0.0
        self._window_start = time.time()
        self.vis_dir = vis_dir
        self.cam_idx = cam_idx
        if self.vis_dir:
            os.makedirs(self.vis_dir, exist_ok=True)

    def stop(self):
        self._stopper.set()

    def _stopped(self):
        return self._stopper.is_set()

    def run(self):
        if not self.engine.load():
            logging.error('%s failed to load pose engine, pipeline stopped.', self.nodename)
            return
        logging.info('%s pipeline started', self.nodename)
        while not self._stopped():
            frame = self.frame_queue.pop(True)
            try:
                if frame.is_eos:
                    self.data_handler.publish('pose', json.dumps({'detections': [], 'EOF': True}))
                    logging.info('%s EOF reached', self.nodename)
                    break

                t0 = time.perf_counter()
                detections = self.engine.infer(frame.image)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self._lat_acc += latency_ms
                self._counter += 1
                self._log_perf(latency_ms)

                payload = {
                    'frame_id': frame.index,
                    'timestamp': frame.monotonic_timestamp,
                    'bbox_format': 'pixel_xywh_center',
                    'keypoint_format': 'pixel_xyc_17',
                    'detections': [
                        {
                            'bbox_xywh': det.bbox_xywh,
                            'bbox_conf': det.bbox_conf,
                            'keypoints': det.keypoints,
                        } for det in detections
                    ]
                }
                self._save_visualization(frame.image, frame.index, detections)
                self.data_handler.publish('pose', json.dumps(payload))
            except Exception as e:
                logging.error('%s infer/publish failed: %s', self.nodename, e)
            finally:
                frame.release()

        logging.info('%s pipeline stopped', self.nodename)

    def _log_perf(self, latency_ms: float):
        now = time.time()
        if now - self._window_start < 1.0:
            return
        fps = self._counter / (now - self._window_start)
        avg = self._lat_acc / max(self._counter, 1)
        logging.info('%s fps=%.2f latency_ms(cur=%.2f avg=%.2f)', self.nodename, fps, latency_ms, avg)
        self._window_start = now
        self._counter = 0
        self._lat_acc = 0.0

    def _letterbox_to_640(self, image):
        src = image
        if src.ndim == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

        h, w = src.shape[:2]
        scale = min(640.0 / max(w, 1), 640.0 / max(h, 1))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((640, 640, 3), dtype=src.dtype)
        pad_x = (640 - nw) // 2
        pad_y = (640 - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y

    def _map_point(self, x, y, scale, pad_x, pad_y):
        return int(round(x * scale + pad_x)), int(round(y * scale + pad_y))

    def _save_visualization(self, image, frame_id: int, detections):
        if not self.vis_dir:
            return

        output_path = os.path.join(self.vis_dir, f"cam{self.cam_idx}_{frame_id:06d}.jpg")
        vis, scale, pad_x, pad_y = self._letterbox_to_640(image)
        if os.path.exists(output_path):
            existing = cv2.imread(output_path)
            if existing is not None and existing.shape[:2] == (640, 640):
                vis = existing

        fluorescent_yellow = (0, 255, 255)
        color_head = (0, 255, 0)
        color_arm = (255, 128, 0)
        color_body = (255, 0, 255)
        color_leg = (0, 165, 255)

        head = {0, 1, 2, 3, 4}
        arms = {5, 6, 7, 8, 9, 10}
        body = {11, 12}
        legs = {13, 14, 15, 16}

        for det in detections:
            x, y, w, h = det.bbox_xywh
            x1, y1 = self._map_point(x - w / 2, y - h / 2, scale, pad_x, pad_y)
            x2, y2 = self._map_point(x + w / 2, y + h / 2, scale, pad_x, pad_y)
            cv2.rectangle(vis, (x1, y1), (x2, y2), fluorescent_yellow, 1)

            mapped_kpts = {}
            for idx, (kx, ky, kc) in enumerate(det.keypoints):
                if kc < 0.3:
                    continue
                px, py = self._map_point(kx, ky, scale, pad_x, pad_y)
                mapped_kpts[idx] = (px, py)
                if idx in head:
                    color = color_head
                elif idx in arms:
                    color = color_arm
                elif idx in body:
                    color = color_body
                elif idx in legs:
                    color = color_leg
                else:
                    color = (255, 255, 255)
                cv2.circle(vis, (px, py), 2, color, -1)

            head_edges = [(0, 1), (0, 2), (1, 3), (2, 4)]
            arm_edges = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6)]
            body_edges = [(5, 11), (6, 12), (11, 12)]
            leg_edges = [(11, 13), (13, 15), (12, 14), (14, 16)]
            for a, b in head_edges:
                if a in mapped_kpts and b in mapped_kpts:
                    cv2.line(vis, mapped_kpts[a], mapped_kpts[b], color_head, 1)
            for a, b in arm_edges:
                if a in mapped_kpts and b in mapped_kpts:
                    cv2.line(vis, mapped_kpts[a], mapped_kpts[b], color_arm, 1)
            for a, b in body_edges:
                if a in mapped_kpts and b in mapped_kpts:
                    cv2.line(vis, mapped_kpts[a], mapped_kpts[b], color_body, 1)
            for a, b in leg_edges:
                if a in mapped_kpts and b in mapped_kpts:
                    cv2.line(vis, mapped_kpts[a], mapped_kpts[b], color_leg, 1)

        cv2.imwrite(output_path, vis)
