import time
from typing import Protocol
import numpy as np
import threading


class FrameProtocol(Protocol):
    is_eos: bool

    @property
    def height(self) -> int: ...

    @property
    def image(self) -> np.ndarray[np.uint8]: ...

    @property
    def index(self) -> int: ...

    @property
    def monotonic_timestamp(self) -> float: ...

    @property
    def timestamp(self) -> float: ...

    @property
    def width(self) -> int: ...


class FrameHandleProtocol(Protocol):
    slot_idx: int
    frame_id: int


class ImageBufferProtocol(Protocol):
    def clear(self) -> None: ...

    def get(self, handle: FrameHandleProtocol) -> FrameProtocol: ...

    def pop(self, blocking: bool = True) -> FrameProtocol: ...

    def pop_handle(self, consumer_id: int, blocking: bool = True) -> FrameHandleProtocol: ...

    def push(self, frame: FrameProtocol) -> None: ...

    def register_consumer(self, name: str) -> int: ...

    def release(self, handle: FrameHandleProtocol) -> None: ...


class FakeFrame:
    def __init__(self, image: np.ndarray, index: int, is_eos=False):
        self._image = image
        self._index = index
        self._is_eos = is_eos
        self._timestamp = time.time()
        self._monotonic_timestamp = time.monotonic()

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def index(self) -> int:
        return self._index

    @property
    def is_eos(self) -> bool:
        return self._is_eos

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def monotonic_timestamp(self) -> float:
        return self._monotonic_timestamp

    @property
    def height(self) -> int:
        return self._image.shape[0]

    @property
    def width(self) -> int:
        return self._image.shape[1]


class FakeImageBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.slots: list[dict] = []
        self.free_slots: list[int] = []
        self.consumer_queues: list[list[dict]] = []
        self.legacy_consumer: int | None = None

    def _ensure_legacy_consumer(self) -> int:
        if self.legacy_consumer is None:
            self.legacy_consumer = self.register_consumer("legacy")
        return self.legacy_consumer

    def push(self, frame):
        with self.cv:
            if not self.consumer_queues:
                return
            if self.free_slots:
                slot_idx = self.free_slots.pop()
                if slot_idx < len(self.slots):
                    self.slots[slot_idx]["frame"] = frame
                    self.slots[slot_idx]["refcnt"] = len(self.consumer_queues)
                else:
                    self.slots.append({"frame": frame, "refcnt": len(self.consumer_queues)})
            else:
                slot_idx = len(self.slots)
                self.slots.append({"frame": frame, "refcnt": len(self.consumer_queues)})
            for queue in self.consumer_queues:
                queue.append({"slot_idx": slot_idx, "frame_id": getattr(frame, "index", 0)})
            self.cv.notify_all()

    def pop(self, blocking=True):
        consumer_id = self._ensure_legacy_consumer()
        handle = self.pop_handle(consumer_id, blocking)
        if handle is None:
            return None
        frame = self.get(handle)
        self.release(handle)
        return frame

    def pop_handle(self, consumer_id: int, blocking: bool = True):
        with self.cv:
            if consumer_id >= len(self.consumer_queues):
                return None
            queue = self.consumer_queues[consumer_id]
            while blocking and not queue:
                self.cv.wait()
            if not queue:
                return None
            return queue.pop(0)

    def get(self, handle):
        if handle is None:
            return None
        with self.cv:
            if handle["slot_idx"] >= len(self.slots):
                return None
            return self.slots[handle["slot_idx"]]["frame"]

    def release(self, handle):
        if handle is None:
            return
        with self.cv:
            slot_idx = handle["slot_idx"]
            if slot_idx >= len(self.slots):
                return
            self.slots[slot_idx]["refcnt"] -= 1
            if self.slots[slot_idx]["refcnt"] <= 0:
                self.free_slots.append(slot_idx)
            self.cv.notify_all()

    def register_consumer(self, name: str):
        with self.cv:
            self.consumer_queues.append([])
            return len(self.consumer_queues) - 1

    def clear(self):
        with self.cv:
            self.slots.clear()
            self.free_slots.clear()
            self.consumer_queues.clear()
            self.legacy_consumer = None
