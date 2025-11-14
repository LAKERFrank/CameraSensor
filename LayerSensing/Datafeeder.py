import threading
import json
import time
from pathlib import Path
from typing import Iterable, List, MutableMapping, Optional

import pandas as pd
import paho.mqtt.client as mqtt

from lib.point import Point, removeOutliers, smooth


class Datafeeder(threading.Thread):
    def __init__(
        self,
        mqttc: mqtt.Client,
        device_name: str,
        filepath: str,
        metapath: Optional[str],
        posepath: Optional[str] = None,
        *,
        pose_playback_speed: float = 1.0,
    ):
        threading.Thread.__init__(self)

        self.mqttc = mqttc
        self.deviceName = device_name
        self.filepath = filepath
        self.metapath = metapath
        self.meta_df = None

        self.posepath = Path(posepath).resolve() if posepath else None
        self.pose_entries: List[MutableMapping] = []
        self.pose_index = 0
        self.pose_first_ts: Optional[float] = None
        self.pose_duration: Optional[float] = None
        self.pose_playback_speed = max(pose_playback_speed, 1e-6)

        if self.posepath is not None:
            self.pose_entries = list(self._load_pose_entries(self.posepath))
            if not self.pose_entries:
                raise ValueError(f"No pose entries were found in {self.posepath}")
            self.pose_first_ts = self._first_pose_timestamp(self.pose_entries)
            self.pose_duration = self._pose_entries_duration(self.pose_entries)

    def run(self):
        tracknet_topic = f"/DATA/{self.deviceName}/SensingLayer/TrackNet"
        pose_topic = (
            f"/DATA/{self.deviceName}/SensingLayer/Pose" if self.pose_entries else None
        )

        df = pd.read_csv(self.filepath)

        if self.metapath:
            self.meta_df = pd.read_csv(self.metapath)

        start = time.time()
        start_ts = float(df.iloc[0].Timestamp)

        for i in range(int(df.shape[0]/10)):
            payload = {"linear": []}
            point_list = []

            sub = df[i*10:i*10+10]

            ts_list = [self._row_ts(row) for _, row in sub.iterrows()]
            t0, t1 = min(ts_list), max(ts_list)
            # payload = {"batch_start_ts": t0, "batch_end_ts": t1, "linear": []}

            for _, row in sub[sub.Visibility == 1].iterrows():
                if self.meta_df is not None and int(row.Frame) in self.meta_df.index:
                    timestamp = self.meta_df.loc[int(row.Frame), "timestamp"]
                else:
                    timestamp = row.Timestamp
                point = Point(
                    fid=row.Frame,
                    timestamp=timestamp,
                    visibility=row.Visibility,
                    x=row.X,
                    y=row.Y,
                    z=row.Z,
                    event=row.Event,
                    speed=0
                )

                # payload["linear"].append(point.toJson())
                point_list.append(point)

            point_list = removeOutliers(point_list)
            # point_list = smooth(point_list)
            for p in point_list:
                payload["linear"].append(p.toJson())

            self._wait_until(start, sub.iloc[0].Timestamp - start_ts, pose_topic)

            self.mqttc.publish(tracknet_topic, json.dumps(payload))

            if pose_topic is not None:
                self._publish_pose_entries(time.time() - start, pose_topic)

            time.sleep(0.1)

        # End of Stream
        self.mqttc.publish(tracknet_topic, json.dumps({"linear": [], "EOF": True}))
        print(f"{tracknet_topic}: EOF")

        if pose_topic is not None:
            self._publish_pose_entries(float("inf"), pose_topic)
            self.mqttc.publish(pose_topic, json.dumps({"EOF": True}))
            print(f"{pose_topic}: EOF")

    def _row_ts(self, row):
        """Prefer the timestamp from the meta file; otherwise use the Timestamp value from the original CSV."""
        if self.meta_df is not None and int(row['Frame']) in self.meta_df.index:
            return float(self.meta_df.loc[int(row.Frame), "timestamp"])
        return float(row['Timestamp'])

    # ------------------------------------------------------------------
    def _load_pose_entries(self, path: Path) -> Iterable[MutableMapping]:
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if isinstance(data, MutableMapping):
                    yield data
                else:
                    raise ValueError("Each JSON line must be an object")
            return

        raw = json.loads(path.read_text(encoding="utf-8"))
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
    def _first_pose_timestamp(entries: List[MutableMapping]) -> Optional[float]:
        for entry in entries:
            ts = Datafeeder._pose_entry_timestamp(entry)
            if ts is not None:
                return ts
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _pose_entry_timestamp(entry: MutableMapping) -> Optional[float]:
        for key in ("monotonic_timestamp", "timestamp"):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    # ------------------------------------------------------------------
    def _pose_entry_elapsed(self, entry: MutableMapping) -> Optional[float]:
        ts = self._pose_entry_timestamp(entry)
        if ts is None or self.pose_first_ts is None:
            return None
        return max(0.0, (ts - self.pose_first_ts) / self.pose_playback_speed)

    # ------------------------------------------------------------------
    @staticmethod
    def _pose_entries_duration(entries: List[MutableMapping]) -> Optional[float]:
        first = Datafeeder._pose_entry_timestamp(entries[0])
        last = Datafeeder._pose_entry_timestamp(entries[-1])
        if first is None or last is None:
            return None
        return max(0.0, last - first)

    # ------------------------------------------------------------------
    def _publish_pose_entries(self, elapsed: float, topic: str) -> None:
        while self.pose_index < len(self.pose_entries):
            entry = self.pose_entries[self.pose_index]
            target_elapsed = self._pose_entry_elapsed(entry)
            if target_elapsed is not None and target_elapsed > elapsed:
                break
            self.mqttc.publish(topic, json.dumps(entry))
            self.pose_index += 1

    # ------------------------------------------------------------------
    def _wait_until(self, start_time: float, target_offset: float, pose_topic: Optional[str]) -> None:
        while True:
            elapsed = time.time() - start_time
            if pose_topic is not None:
                self._publish_pose_entries(elapsed, pose_topic)

            remaining = target_offset - elapsed
            if remaining <= 0:
                break
            time.sleep(min(0.1, max(0.0, remaining)))
