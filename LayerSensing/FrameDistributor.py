import logging
import queue
import threading
from dataclasses import dataclass

CONSUMER_TRACKNET = 1
CONSUMER_POSE = 2


@dataclass
class SharedFrameState:
    frame: object
    need_mask: int
    released_mask: int = 0

    def need_pose(self) -> bool:
        return bool(self.need_mask & CONSUMER_POSE)

    def ref_count(self) -> int:
        return int(bool(self.need_mask & CONSUMER_TRACKNET and not self.released_mask & CONSUMER_TRACKNET)) + \
            int(bool(self.need_mask & CONSUMER_POSE and not self.released_mask & CONSUMER_POSE))


class SharedFrameHandle:
    def __init__(self, distributor: 'FrameDistributor', state: SharedFrameState, consumer_mask: int):
        self._distributor = distributor
        self._state = state
        self._consumer_mask = consumer_mask
        self._released = False

    @property
    def image(self):
        return self._state.frame.image

    @property
    def index(self):
        return self._state.frame.index

    @property
    def monotonic_timestamp(self):
        return self._state.frame.monotonic_timestamp

    @property
    def timestamp(self):
        return self._state.frame.timestamp

    @property
    def is_eos(self):
        return self._state.frame.is_eos

    def release(self):
        if self._released:
            return
        self._released = True
        self._distributor.release(self._state, self._consumer_mask)


class ConsumerFrameQueue:
    def __init__(self, distributor: 'FrameDistributor', consumer_mask: int):
        self._queue = queue.Queue(maxsize=distributor.consumer_queue_size)
        self._distributor = distributor
        self._consumer_mask = consumer_mask

    def pop(self, blocking=True):
        state = self._queue.get(block=blocking)
        return SharedFrameHandle(self._distributor, state, self._consumer_mask)

    def push_state(self, state: SharedFrameState):
        self._queue.put_nowait(state)

    def pop_nowait_state(self):
        return self._queue.get_nowait()

    def empty(self):
        return self._queue.empty()


class FrameDistributor(threading.Thread):
    def __init__(self, image_buffer, pose_stride: int = 4, pose_queue_size: int = 8, consumer_queue_size: int = 256):
        super().__init__(name='FrameDistributor', daemon=True)
        self.image_buffer = image_buffer
        self.pose_stride = pose_stride
        self.consumer_queue_size = consumer_queue_size
        self.tracknet_queue = ConsumerFrameQueue(self, CONSUMER_TRACKNET)
        self.pose_queue = ConsumerFrameQueue(self, CONSUMER_POSE)
        self.pose_queue._queue = queue.Queue(maxsize=pose_queue_size)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._active_tracknet = False
        self._active_pose = False
        self.pose_drop_count = 0

    def activate_tracknet(self, active: bool):
        with self._lock:
            self._active_tracknet = active

    def activate_pose(self, active: bool):
        with self._lock:
            self._active_pose = active

    def run(self):
        while not self._stop_event.is_set():
            frame = self.image_buffer.pop(True)
            with self._lock:
                need_tracknet = self._active_tracknet
                need_pose = self._active_pose and (not frame.is_eos) and (frame.index % self.pose_stride == 0)

            if not need_tracknet and not need_pose:
                if frame.is_eos:
                    logging.info('FrameDistributor got EOS while idle.')
                continue

            need_mask = (CONSUMER_TRACKNET if need_tracknet else 0) | (CONSUMER_POSE if need_pose else 0)
            state = SharedFrameState(frame=frame, need_mask=need_mask)

            if need_tracknet:
                self.tracknet_queue.push_state(state)
            if need_pose:
                self._push_pose_with_drop(state)

            logging.debug(
                'Distribute frame=%s need_pose=%s ref_count=%s',
                frame.index if not frame.is_eos else 'eos',
                need_pose,
                state.ref_count()
            )

            if frame.is_eos:
                break

    def _push_pose_with_drop(self, state: SharedFrameState):
        try:
            self.pose_queue.push_state(state)
        except queue.Full:
            dropped = self.pose_queue.pop_nowait_state()
            self.release(dropped, CONSUMER_POSE)
            self.pose_drop_count += 1
            logging.warning('Pose queue full, drop oldest frame_id=%s (drop_count=%s)', dropped.frame.index, self.pose_drop_count)
            self.pose_queue.push_state(state)

    def release(self, state: SharedFrameState, consumer_mask: int):
        with self._lock:
            if not (state.need_mask & consumer_mask):
                logging.warning('release mismatch frame_id=%s consumer=%s need_mask=%s', state.frame.index, consumer_mask, state.need_mask)
                return
            if state.released_mask & consumer_mask:
                logging.warning('double release frame_id=%s consumer=%s', state.frame.index, consumer_mask)
                return
            state.released_mask |= consumer_mask
            logging.debug(
                'Release frame=%s consumer=%s ref_count=%s released_mask=%s',
                state.frame.index,
                consumer_mask,
                state.ref_count(),
                state.released_mask,
            )

    def stop(self):
        self._stop_event.set()
