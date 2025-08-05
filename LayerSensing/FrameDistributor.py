import threading
import time
from LayerCamera.CameraSystemC.recorder_module import ImageBuffer

class FrameDistributor(threading.Thread):
    """Fetch frames from a source buffer and distribute them to TrackNet and Pose buffers.

    Pose only receives frames whose index is divisible by ``pose_interval``.
    """

    def __init__(self, src_buf: ImageBuffer, tracknet_buf: ImageBuffer,
                 pose_buf: ImageBuffer, pose_fps: int = 30, src_fps: int = 120):
        """Initialize distributor with target pose FPS.

        Args:
            src_buf (ImageBuffer): Source frames.
            tracknet_buf (ImageBuffer): Destination buffer for TrackNet.
            pose_buf (ImageBuffer): Destination buffer for Pose.
            pose_fps (int, optional): Desired pose frames per second, e.g. 120, 60, 40, 30, 20.
            src_fps (int, optional): Source camera FPS. Defaults to 120.
        """
        super().__init__()
        self.src_buf = src_buf
        self.tracknet_buf = tracknet_buf
        self.pose_buf = pose_buf
        self.pose_interval = max(1, src_fps // pose_fps)
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
