import json
import logging
import os
import threading
import time

import cv2

from LayerSensing.Pose.PoseEngine import TensorRTPoseEngine


class PoseMqtt(threading.Thread):
    def __init__(self, nodename, data_handler, frame_queue, engine_path: str, vis_dir: str = None):
        super().__init__(name=nodename)
        self.nodename = nodename
        self.data_handler = data_handler
        self.frame_queue = frame_queue
        self.engine = TensorRTPoseEngine(engine_path=engine_path)
        self.vis_dir = vis_dir
        self._stopper = threading.Event()
        self._counter = 0
        self._lat_acc = 0.0
        self._window_start = time.time()

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
                self.data_handler.publish('pose', json.dumps(payload))
                self._save_visualization(frame, detections)
            except Exception as e:
                logging.error('%s infer/publish failed: %s', self.nodename, e)
            finally:
                frame.release()

        logging.info('%s pipeline stopped', self.nodename)

    def _save_visualization(self, frame, detections):
        if not self.vis_dir:
            return

        image = frame.image
        if image is None:
            return

        if image.ndim == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            vis = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        for det in detections:
            cx, cy, w, h = det.bbox_xywh
            x1 = int(max(cx - w / 2.0, 0))
            y1 = int(max(cy - h / 2.0, 0))
            x2 = int(min(cx + w / 2.0, vis.shape[1] - 1))
            y2 = int(min(cy + h / 2.0, vis.shape[0] - 1))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for x, y, kconf in det.keypoints:
                if kconf <= 0:
                    continue
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 255), -1)

        os.makedirs(self.vis_dir, exist_ok=True)
        save_path = os.path.join(self.vis_dir, f'{frame.index}.png')
        cv2.imwrite(save_path, vis)

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
