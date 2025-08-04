import threading
import json
import time
import pandas as pd
import paho.mqtt.client as mqtt

from lib.point import Point

class PoseDatafeeder(threading.Thread):
    """Feed pose CSV results via MQTT similar to TrackNet Datafeeder."""

    def __init__(self, mqttc: mqtt.Client, device_name: str, filepath: str, metapath: str | None = None):
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

        for frame_id, rows in df.groupby('Frame'):
            payload = {"linear": []}
            for _, row in rows.iterrows():
                if self.meta_df is not None and row.Frame in self.meta_df.index:
                    ts = self.meta_df.loc[row.Frame, 'monotonic_timestamp']
                else:
                    ts = row.Timestamp
                point = Point(
                    fid=row.Frame,
                    timestamp=ts,
                    visibility=1,
                    x=row['kp1_x'],
                    y=row['kp1_y'],
                    z=0,
                    event=0,
                    speed=0,
                )
                payload['linear'].append(point.toJson())
            while (time.time() - start) <= (rows.iloc[0].Timestamp - start_ts):
                time.sleep(0.01)
            self.mqttc.publish(pose_topic, json.dumps(payload))

        self.mqttc.publish(pose_topic, json.dumps({"linear": [], "EOF": True}))
        print(f"{pose_topic}: EOF")
