import argparse
import json
import paho.mqtt.client as mqtt


def on_message(_client, _userdata, msg):
    try:
        payload = json.loads(msg.payload)
    except json.JSONDecodeError:
        payload = msg.payload.decode(errors="ignore")
    print(f"[PoseData] {msg.topic}: {payload}")


def main():
    parser = argparse.ArgumentParser(description="Receive pose data from MQTT and log it")
    parser.add_argument("--broker", default="localhost", help="MQTT broker address")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--device", required=True, help="Device name used in MQTT topics")
    args = parser.parse_args()

    topic = f"/DATA/{args.device}/LayerSensing/Pose"
    mqttc = mqtt.Client()
    mqttc.on_message = on_message
    mqttc.connect(args.broker, args.port)
    mqttc.subscribe(topic)

    print(f"Listening on topic: {topic}")
    mqttc.loop_forever()


if __name__ == "__main__":
    main()
