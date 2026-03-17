import csv
import json
import logging
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
            writer.writerow([
                'frame_id', 'timestamp', 'det_index',
                'bbox_cx', 'bbox_cy', 'bbox_w', 'bbox_h', 'bbox_conf',
                'kp_index', 'kp_x', 'kp_y', 'kp_conf'
            ])

    def _append_pose_csv(self, payload):
        rows = []
        frame_id = payload.get('frame_id')
        timestamp = payload.get('timestamp')
        for det_idx, det in enumerate(payload.get('detections', [])):
            bbox = det.get('bbox_xywh', [None, None, None, None])
            bbox_conf = det.get('bbox_conf')
            keypoints = det.get('keypoints', [])
            if not keypoints:
                rows.append([frame_id, timestamp, det_idx, *bbox[:4], bbox_conf, None, None, None, None])
                continue
            for kp_idx, kp in enumerate(keypoints):
                kp_vals = list(kp) if isinstance(kp, (list, tuple)) else [None, None, None]
                if len(kp_vals) < 3:
                    kp_vals.extend([None] * (3 - len(kp_vals)))
                rows.append([frame_id, timestamp, det_idx, *bbox[:4], bbox_conf, kp_idx, kp_vals[0], kp_vals[1], kp_vals[2]])

        if not rows:
            rows.append([frame_id, timestamp, None, None, None, None, None, None, None, None, None, None])

        with open(self.output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

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
                self._append_pose_csv(payload)
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
