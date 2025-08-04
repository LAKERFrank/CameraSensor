import threading
import json
import time
from typing import Optional

import pandas as pd
import paho.mqtt.client as mqtt

class PoseDatafeeder(threading.Thread):
    """Feed pose CSV results via MQTT similar to TrackNet Datafeeder."""

    def __init__(self, mqttc: mqtt.Client, device_name: str, filepath: str, metapath: Optional[str] = None):
        super().__init__()
        self.mqttc = mqttc
        self.deviceName = device_name
        self.filepath = filepath
        self.metapath = metapath
        self.meta_df = None

    def run(self):
        pose_topic = f"/DATA/{self.deviceName}/LayerSensing/Pose"
        df = pd.read_csv(self.filepath)
        if self.metapath:
            # index the meta csv by frame so lookups by frame id are O(1)
            self.meta_df = pd.read_csv(self.metapath, index_col=0)
        start = time.time()
        start_ts = float(df.iloc[0].Timestamp)

        for _, row in df.iterrows():
            if self.meta_df is not None and row.Frame in self.meta_df.index:
                ts = self.meta_df.loc[row.Frame, 'monotonic_timestamp']
            else:
                ts = row.Timestamp
            payload = {
                "id": int(row.Frame),
                "timestamp": ts,
                "bbox_x1": row["bbox_x1"],
                "bbox_y1": row["bbox_y1"],
                "bbox_x2": row["bbox_x2"],
                "bbox_y2": row["bbox_y2"],
            }
            for i in range(1, 18):
                payload[f"kp{i}_x"] = row[f"kp{i}_x"]
                payload[f"kp{i}_y"] = row[f"kp{i}_y"]

            while (time.time() - start) <= (row.Timestamp - start_ts):
                time.sleep(0.01)
            self.mqttc.publish(pose_topic, json.dumps(payload))

        self.mqttc.publish(pose_topic, json.dumps({"EOF": True}))
        print(f"{pose_topic}: EOF")
