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
        tracknet_topic = f"/DATA/{self.deviceName}/LayerSensing/TrackNet"

        df = pd.read_csv(self.filepath)
        
        if self.metapath:
            self.meta_df = pd.read_csv(self.metapath)

        start = time.time()
        start_ts = float(df.iloc[0].Timestamp)

        for i in range(int(df.shape[0]/10)):

            payload = {"linear": []}
            point_list = []

            sub = df[i*10:i*10+10]
            for _, row in sub[sub.Visibility == 1].iterrows():
                if self.meta_df is not None and row.Frame in self.meta_df.index:
                    timestamp = self.meta_df.loc[row.Frame, "monotonic_timestamp"]
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

        # End of Stream
        self.mqttc.publish(tracknet_topic, json.dumps({"linear": [], "EOF": True}))
        print(f"{tracknet_topic}: EOF")
