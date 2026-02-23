import argparse
import json
import signal
import sys
from datetime import datetime

import paho.mqtt.client as mqtt

from lib.common import ROOTDIR, loadConfig

BLUE = "\033[94m"
ORANGE = "\033[38;5;208m"
GRAY = "\033[90m"
RESET = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline content test receiver for TrackNet / Pose MQTT stream"
    )
    parser.add_argument("--broker", type=str, default=None, help="MQTT broker host")
    parser.add_argument("--port", type=int, default=None, help="MQTT broker port")
    parser.add_argument(
        "--device",
        type=str,
        default="CameraReader_0",
        help="Target device name in MQTT topic, e.g. CameraReader_0",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON payload",
    )
    return parser.parse_args()


def decode_payload(payload: bytes):
    try:
        return json.loads(payload)
    except Exception:
        try:
            return payload.decode("utf-8", errors="replace")
        except Exception:
            return str(payload)


def format_payload(data, pretty: bool) -> str:
    if isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False, indent=2 if pretty else None)
    return str(data)


def main():
    args = parse_args()

    cfg = loadConfig(f"{ROOTDIR}/config")
    broker = args.broker or cfg["Project"]["mqtt_broker"]
    port = args.port or int(cfg["Project"]["mqtt_port"])

    tracknet_topic = f"/DATA/{args.device}/SensingLayer/TrackNet"
    pose_topic = f"/DATA/{args.device}/SensingLayer/Pose"

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def on_connect(mqtt_client, userdata, flags, reason_code, properties):
        print(f"Connected MQTT broker {broker}:{port}, reason_code={reason_code}")
        mqtt_client.subscribe(tracknet_topic, qos=0)
        mqtt_client.subscribe(pose_topic, qos=0)
        print(f"Subscribe: {tracknet_topic}")
        print(f"Subscribe: {pose_topic}")

    def on_message(mqtt_client, userdata, msg):
        data = decode_payload(msg.payload)
        text = format_payload(data, args.pretty)
        timestamp = datetime.now().strftime("%H:%M:%S")

        if msg.topic.endswith("/TrackNet"):
            print(f"{BLUE}[{timestamp}] [TRACKNET]{RESET} {text}")
        elif msg.topic.endswith("/Pose"):
            print(f"{ORANGE}[{timestamp}] [POSE]{RESET} {text}")
        else:
            print(f"{GRAY}[{timestamp}] [{msg.topic}]{RESET} {text}")

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker, port)
    client.loop_start()

    stop = {"value": False}

    def handle_signal(signum, frame):
        stop["value"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Offline content test receiver started. Press Ctrl+C to stop.")
    try:
        while not stop["value"]:
            signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
        print("Stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
