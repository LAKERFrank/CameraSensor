import json
import logging
import threading
import time

from LayerSensing.Pose.PoseEngine import TensorRTPoseEngine


class PoseMqtt(threading.Thread):
    def __init__(self, nodename, data_handler, frame_queue, engine_path: str):
        super().__init__(name=nodename)
        self.nodename = nodename
        self.data_handler = data_handler
        self.frame_queue = frame_queue
        self.engine = TensorRTPoseEngine(engine_path=engine_path)
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
