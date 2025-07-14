import threading
import time
from typing import List
from LayerCamera.CameraSystemC.recorder_module import ImageBuffer, Frame

class BufferDistributor(threading.Thread):
    """Duplicate frames from a source ImageBuffer to multiple destination ImageBuffers."""
    def __init__(self, src_buf: ImageBuffer, dst_bufs: List[ImageBuffer]):
        super().__init__()
        self.src_buf = src_buf
        self.dst_bufs = dst_bufs
        self._stopper = threading.Event()

    def stop(self):
        self._stopper.set()

    def _stopped(self):
        return self._stopper.is_set()

    def run(self):
        while not self._stopped():
            frame = self.src_buf.pop(False)
            if frame is None:
                time.sleep(0.001)
                continue
            for buf in self.dst_bufs:
                buf.push(frame)
            if frame.is_eos:
                break
