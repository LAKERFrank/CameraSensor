import threading
import time
from LayerCamera.CameraSystemC.recorder_module import ImageBuffer

class FrameDistributor(threading.Thread):
    """Fetch frames from a source buffer and distribute them to TrackNet and Pose buffers.

    Pose only receives frames whose index is divisible by ``pose_interval``.
    """

    def __init__(self, src_buf: ImageBuffer, tracknet_buf: ImageBuffer,
                 pose_buf: ImageBuffer, pose_interval: int = 4):
        super().__init__()
        self.src_buf = src_buf
        self.tracknet_buf = tracknet_buf
        self.pose_buf = pose_buf
        self.pose_interval = pose_interval
        self._stopper = threading.Event()

    def stop(self):
        self._stopper.set()

    def _stopped(self) -> bool:
        return self._stopper.is_set()

    def run(self):
        while not self._stopped():
            frame = self.src_buf.pop(False)
            if frame is None:
                time.sleep(0.001)
                continue
            self.tracknet_buf.push(frame)
            if frame.index % self.pose_interval == 0:
                self.pose_buf.push(frame)
            if frame.is_eos:
                break
