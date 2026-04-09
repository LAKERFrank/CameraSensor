import json
import logging
import threading
import time
import os
import math

from LayerSensing.Pose.PoseEngine import TensorRTPoseEngine


class PoseMqtt(threading.Thread):
    def __init__(self, nodename, data_handler, frame_queue, batch_size: int, engine_path: str, output_json: str):
        super().__init__(name=nodename)
        self.nodename = nodename
        self.data_handler = data_handler
        self.frame_queue = frame_queue
        self.batch_size = max(int(batch_size), 1)
        self.engine = TensorRTPoseEngine(engine_path=engine_path)
        self.output_json = output_json
        self._stopper = threading.Event()
        self._counter = 0
        self._lat_acc = 0.0
        self._window_start = time.time()

    def _ensure_json_output(self):
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
        if not os.path.exists(self.output_json):
            open(self.output_json, 'w').close()

    def _append_pose_json(self, payload):
        with open(self.output_json, 'a') as jsonfile:
            jsonfile.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def stop(self):
        self._stopper.set()

    def _stopped(self):
        return self._stopper.is_set()

    def run(self):
        pending_frames = []
        try:
            self._ensure_json_output()
            if not self.engine.load():
                logging.error('%s failed to load pose engine, pipeline stopped.', self.nodename)
                return
            logging.info('%s pipeline started', self.nodename)
            while not self._stopped():
                frame = self.frame_queue.pop(True)
                try:
                    if frame.is_eos:
                        if pending_frames:
                            self._process_batch(pending_frames)
                            pending_frames = []
                        payload = {'frame_id': frame.index, 'timestamp': frame.monotonic_timestamp, 'detection': [], 'EOF': True}
                        self.data_handler.publish('pose', json.dumps(payload))
                        self._append_pose_json(payload)
                        logging.info('%s EOF reached', self.nodename)
                        break

                    if self.batch_size == 1:
                        self._process_batch([frame])
                        continue

                    pending_frames.append(frame)
                    if len(pending_frames) >= self.batch_size:
                        self._process_batch(pending_frames)
                        pending_frames = []
                except Exception as e:
                    logging.error('%s infer/publish failed: %s', self.nodename, e)
                    frame.release()
        finally:
            for pending in pending_frames:
                pending.release()
            self.engine.unload()
            logging.info('%s pipeline stopped', self.nodename)

    def _process_batch(self, frames):
        try:
            t0 = time.perf_counter()
            batch_detections = self.engine.infer_batch([f.image for f in frames])
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._lat_acc += latency_ms
            self._counter += len(frames)
            self._log_perf(latency_ms)

            for idx, frame in enumerate(frames):
                detections = batch_detections[idx] if idx < len(batch_detections) else []
                payload = self._build_payload(frame, detections)
                self.data_handler.publish('pose', json.dumps(payload))
                self._append_pose_json(payload)
        finally:
            for frame in frames:
                frame.release()

    def _build_payload(self, frame, detections):
        h, w = frame.image.shape[:2]
        payload = {
            'frame_id': frame.index,
            'timestamp': frame.monotonic_timestamp,
            'detection': [],
        }

        top_detections = (detections or [])[:2]
        for det in top_detections:
            cx, cy, bw, bh = det.bbox_xywh
            bbox = [
                round(float(min(max(cx / w, 0.0), 1.0)) if w else 0.0, 5),
                round(float(min(max(cy / h, 0.0), 1.0)) if h else 0.0, 5),
                round(float(min(max(bw / w, 0.0), 1.0)) if w else 0.0, 5),
                round(float(min(max(bh / h, 0.0), 1.0)) if h else 0.0, 5),
                round(float(det.bbox_conf), 3),
            ]
            kpts = []
            for x, y, _ in det.keypoints[:17]:
                kpts.append(round(float(min(max(x / w, 0.0), 1.0)) if w else 0.0, 5))
                kpts.append(round(float(min(max(y / h, 0.0), 1.0)) if h else 0.0, 5))
            while len(kpts) < 34:
                kpts.extend([0.0, 0.0])

            payload['detection'].append({
                'bbox': bbox,
                'kpts': kpts,
            })

        return payload

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
