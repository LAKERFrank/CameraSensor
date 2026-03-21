import csv
import json
import logging
import math
import os
import threading
import time

from LayerSensing.Pose.PoseEngine import TensorRTPoseEngine


class PoseMqtt(threading.Thread):
    def __init__(self, nodename, data_handler, frame_queue, engine_path: str, output_csv: str):
        super().__init__(name=nodename)
        self.nodename = nodename
        self.data_handler = data_handler
        self.frame_queue = frame_queue
        self.engine = TensorRTPoseEngine(engine_path=engine_path)
        self.output_csv = output_csv
        self._stopper = threading.Event()
        self._counter = 0
        self._lat_acc = 0.0
        self._window_start = time.time()

    def _ensure_csv_header(self):
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        if os.path.exists(self.output_csv) and os.path.getsize(self.output_csv) > 0:
            return
        with open(self.output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['frame_id', 'timestamp', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'bbox_conf']
            for kp_idx in range(17):
                header.extend([f'kpt{kp_idx}_x', f'kpt{kp_idx}_y'])
            writer.writerow(header)

    def _normalize_coord(self, value, size):
        if value is None or size in (None, 0):
            return None
        try:
            value = float(value)
            size = float(size)
        except (TypeError, ValueError):
            return None
        if math.isnan(value) or math.isnan(size):
            return None
        return value / size

    def _append_pose_csv(self, payload, frame_width, frame_height):
        frame_id = payload.get('frame_id')
        timestamp = payload.get('timestamp')
        detections = payload.get('detections', [])

        best_det = None
        best_conf = float('-inf')
        for det in detections:
            conf = det.get('bbox_conf')
            try:
                score = float(conf)
            except (TypeError, ValueError):
                score = float('-inf')
            if best_det is None or score > best_conf:
                best_det = det
                best_conf = score

        if best_det is None:
            row = [frame_id, timestamp, None, None, None, None, None, *([None] * 34)]
        else:
            bbox = list(best_det.get('bbox_xywh', [None, None, None, None]))
            if len(bbox) < 4:
                bbox.extend([None] * (4 - len(bbox)))
            bbox = bbox[:4]
            norm_bbox = [
                self._normalize_coord(bbox[0], frame_width),
                self._normalize_coord(bbox[1], frame_height),
                self._normalize_coord(bbox[2], frame_width),
                self._normalize_coord(bbox[3], frame_height),
            ]
            bbox_conf = best_det.get('bbox_conf')

            flat_kpts = []
            keypoints = best_det.get('keypoints', [])
            for kp_idx in range(17):
                if kp_idx < len(keypoints) and isinstance(keypoints[kp_idx], (list, tuple)):
                    kp = list(keypoints[kp_idx])
                    x = self._normalize_coord(kp[0] if len(kp) > 0 else None, frame_width)
                    y = self._normalize_coord(kp[1] if len(kp) > 1 else None, frame_height)
                else:
                    x, y = None, None
                flat_kpts.extend([x, y])

            row = [frame_id, timestamp, *norm_bbox, bbox_conf, *flat_kpts]

        with open(self.output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    def stop(self):
        self._stopper.set()

    def _stopped(self):
        return self._stopper.is_set()

    def run(self):
        self._ensure_csv_header()
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
                self._append_pose_csv(payload, frame.width, frame.height)
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
