"""Utilities to replay recorded pose detections over MQTT."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Iterable, List, MutableMapping, Optional


LOGGER = logging.getLogger(__name__)


class PoseDatafeeder(threading.Thread):
    """Replay pose detections from disk and publish them to MQTT.

    The feeder mirrors the behaviour of the realtime pose worker so that
    the Content layer can be exercised without a live camera feed.  The
    input file must contain JSON objects describing pose detections.  Both
    JSON arrays and newline-delimited JSON (``.jsonl``/``.ndjson``) files are
    supported.  Each entry is expected to follow the schema produced by
    :class:`LayerSensing.Pose.pose_worker.PoseWorker`.
    """

    def __init__(
        self,
        data_handler,
        device_name: str,
        filepath: str,
        *,
        playback_speed: float = 1.0,
    ) -> None:
        super().__init__(daemon=True)
        self._data_handler = data_handler
        self._device_name = device_name
        self._filepath = Path(filepath)
        self._playback_speed = max(playback_speed, 1e-6)
        self._stop_event = threading.Event()

        self._entries: List[MutableMapping] = list(self._load_entries())
        if not self._entries:
            raise ValueError(f"No pose entries were found in {self._filepath}")

        self.entry_count: int = len(self._entries)
        self.duration: Optional[float] = self._estimate_duration(self._entries)

    # ------------------------------------------------------------------
    def _load_entries(self) -> Iterable[MutableMapping]:
        suffix = self._filepath.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            content = self._filepath.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if isinstance(data, MutableMapping):
                    yield data
                else:
                    raise ValueError("Each JSON line must be an object")
            return

        # Default to a standard JSON payload (list or object with a list).
        raw = json.loads(self._filepath.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, MutableMapping):
                    yield item
                else:
                    raise ValueError("Pose entry must be a JSON object")
            return

        if isinstance(raw, MutableMapping):
            for key in ("frames", "detections", "data"):
                values = raw.get(key)
                if isinstance(values, list):
                    for item in values:
                        if isinstance(item, MutableMapping):
                            yield item
                        else:
                            raise ValueError("Pose entry must be a JSON object")
                    return
        raise ValueError(
            "Unsupported pose data format; provide a JSON array or JSONL file"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_duration(entries: List[MutableMapping]) -> Optional[float]:
        def _extract_ts(entry: MutableMapping) -> Optional[float]:
            for key in ("monotonic_timestamp", "timestamp"):
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            return None

        start = _extract_ts(entries[0])
        end = _extract_ts(entries[-1])
        if start is None or end is None:
            return None
        return max(0.0, end - start)

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - exercised via integration
        topic_name = "pose"
        playback_start = time.time()
        first_ts = self._first_timestamp(self._entries)

        for entry in self._entries:
            if self._stop_event.is_set():
                break

            if not self._wait_for_schedule(entry, first_ts, playback_start):
                break
            self._publish_entry(topic_name, entry)
            LOGGER.debug(
                "[PoseDatafeeder] %s frame=%s detections=%d",
                self._device_name,
                entry.get("frame_index"),
                len(entry.get("detections", []) or []),
            )

        self._publish_entry(topic_name, {"EOF": True})
        LOGGER.info("Pose datafeeder for %s finished", self._device_name)

    # ------------------------------------------------------------------
    def _first_timestamp(self, entries: List[MutableMapping]) -> Optional[float]:
        for entry in entries:
            for key in ("monotonic_timestamp", "timestamp"):
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        return None

    # ------------------------------------------------------------------
    def _wait_for_schedule(
        self,
        entry: MutableMapping,
        first_ts: Optional[float],
        playback_start: float,
    ) -> bool:
        if first_ts is None:
            return not self._stop_event.is_set()

        for key in ("monotonic_timestamp", "timestamp"):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                target_elapsed = (float(value) - first_ts) / self._playback_speed
                now_elapsed = time.time() - playback_start
                wait_time = target_elapsed - now_elapsed
                if wait_time > 0:
                    if self._stop_event.wait(wait_time):
                        return False
                return not self._stop_event.is_set()
        return not self._stop_event.is_set()
    # ------------------------------------------------------------------
    def _publish_entry(self, topic_name: str, payload) -> None:
        if self._stop_event.is_set():
            return
        if isinstance(payload, str):
            message = payload
        else:
            message = json.dumps(payload)
        self._data_handler.publish(topic_name, message)


__all__ = ["PoseDatafeeder"]

