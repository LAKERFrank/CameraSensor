import threading
import json
import time
import paho.mqtt.client as mqtt
import logging
import math
from lib.MqttClient import MqttClient
from lib.common import ROOTDIR, loadConfig
from lib.GracefulKiller import GracefulKiller

from LayerCamera.camera.RpcCamera import RpcCamera

# 誤差容忍值 (單位:ns)
OFFSET_THRESHOLD = 800_000 # 0.8 ms
# 檢查間隔 (單位:秒)
CHECK_PERIOD = 10

DEBUG = False

class Operator:
    def __init__(self, name:str, mqttc:mqtt.Client):
        self.name = name
        self.mqttc = mqttc
        self.rpcCamera = RpcCamera(name, mqttc)
        self.timestamp:float = 0
        self.dt:float = 0
        self.offset:float = 0

        self.lock = threading.Lock()

        # Disable Heartbeat -> use query
        #self._setupCallback()

    def _setupCallback(self):
        topic = f"/DATA/{self.name}/CameraLayer/Heartbeat"
        self.mqttc.message_callback_add(
            topic,
            self._metricUpdate
        )
        self.mqttc.subscribe(topic)

    def _metricUpdate(self, client, userdata, msg):
        data = json.loads(msg.payload)

        if data:
            with self.lock:
                self.timestamp = data['kf_timestamp']
                self.dt = data['kf_dt']

        logging.info(f"[{self.name}] Received heartbeat "
                     f"with timestamp={self.timestamp}, dt={self.dt}")

    def query(self):
        data = self.rpcCamera.getMetricData()
        with self.lock:
            self.timestamp = data['kf_timestamp']
            self.dt = data['kf_dt']

        logging.info(f"[{self.name}] query "
                     f"with timestamp={self.timestamp}, dt={self.dt}")

    def getTimestampDt(self) -> 'tuple[float, float]':
        with self.lock:
            return (self.timestamp, self.dt)

    def calculateOffset(self, ref_timestamp, ref_dt):
        with self.lock:
            timestamp, dt = self.timestamp, self.dt

        # align timestamp with ref_timestamp
        frame_diff = round((ref_timestamp - timestamp) / dt)
        aligned_timestamp = timestamp + frame_diff * dt
        self.offset = aligned_timestamp - ref_timestamp

        logging.info(f"[{self.name}] offset {self.offset*1000:+.3f}ms")

        if abs(self.offset*1000_000_000) > OFFSET_THRESHOLD:
            # convert timestamp (sec to ns)
            resync_timestamp_ns = int(ref_timestamp * 1000_000_000)
            resync_dt_ns = int(ref_dt * 1000_000_000)
            self.rpcCamera.resync(resync_timestamp_ns, resync_dt_ns)
            logging.info(f"[{self.name}] triggered resync timestamp {resync_timestamp_ns} ({resync_timestamp_ns/1e9 - time.time():+.6f})")

            if DEBUG: # show offset after resync
                # update with newest timestamp, dt (wait for resync done)
                time.sleep(1)
                self.query()

                # show new_offset
                with self.lock:
                    timestamp, dt = self.timestamp, self.dt
                aligned_timestamp = timestamp + round((ref_timestamp - timestamp) / dt) * dt
                new_offset = aligned_timestamp - ref_timestamp
                logging.info(f"[{self.name}] offset after resync ({self.offset*1000:+.3f} -> {new_offset*1000:+.3f} ms)")

class CameraSync(threading.Thread):
    def __init__(self, mqttc:mqtt.Client, camera_names:'list[str]'=[]):
        super().__init__()
        self.mqttc = mqttc
        self.stopEvent = threading.Event()

        self.ops = [Operator(name, self.mqttc) for name in camera_names]

    def stop(self):
        self.stopEvent.set()

    def _chooseRefCamera(self):
        return 0
    def run(self):
        logging.info(f"started")

        if len(self.ops) < 2:
            # 小於兩隻相機不需要同步
            return

        while not self.stopEvent.isSet():

            # queryAll
            for op in self.ops:
                op.query()

            logging.info("==="*10)
            logging.info(f"check")

            ref_idx = self._chooseRefCamera()
            logging.info(f"Chosen ref camera index={ref_idx}")
            ref_timestamp, ref_dt = self.ops[ref_idx].getTimestampDt()

            # 向當前時間點對齊
            new_ref_timestamp = ref_timestamp + math.ceil((time.time() - ref_timestamp) / ref_dt)*ref_dt

            # calculate other cameras offset and resync
            for i in range(len(self.ops)):
                if i == ref_idx:
                    continue
                self.ops[i].calculateOffset(new_ref_timestamp, ref_dt)

            time.sleep(CHECK_PERIOD)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][SyncThread] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger().setLevel(logging.INFO)

    cfg = loadConfig(f"{ROOTDIR}/config")

    broker_ip = cfg["Project"]["mqtt_broker"]
    broker_port = int(cfg["Project"]["mqtt_port"])

    client = MqttClient()
    client.start(broker_ip, broker_port)

    camera_names = [
        'tcam-08324018',
        'tcam-38324265',
        'tcam-44224012',
        'tcam-44224013',
    ]

    cs = CameraSync(client.mqttc, camera_names)
    cs.start()

    GracefulKiller().wait()

    cs.stop()
    cs.join()

    client.stop()
