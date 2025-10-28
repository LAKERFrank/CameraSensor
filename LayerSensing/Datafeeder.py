import threading
import json
import time
import pandas as pd
import paho.mqtt.client as mqtt

from lib.point import Point, removeOutliers, smooth

class Datafeeder(threading.Thread):
    def __init__(self, mqttc:mqtt.Client, device_name:str, filepath:str, metapath:str):
        threading.Thread.__init__(self)

        self.mqttc = mqttc
        self.deviceName = device_name
        self.filepath = filepath
        self.metapath = metapath
        self.meta_df = None

    def run(self):
        tracknet_topic = f"/DATA/{self.deviceName}/SensingLayer/TrackNet"

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

            while (time.time() - start) <= (sub.iloc[0].Timestamp - start_ts):
                time.sleep(0.2)

            self.mqttc.publish(tracknet_topic, json.dumps(payload))
            # print(tracknet_topic)
            # print(json.dumps(payload))

            time.sleep(0.1)

        # End of Stream
        self.mqttc.publish(tracknet_topic, json.dumps({"linear": [], "EOF": True}))
        print(f"{tracknet_topic}: EOF")

    def _row_ts(self, row):
        """Prefer the timestamp from the meta file; otherwise use the Timestamp value from the original CSV."""
        if self.meta_df is not None and int(row['Frame']) in self.meta_df.index:
            return float(self.meta_df.loc[int(row.Frame), "timestamp"])
        return float(row['Timestamp'])