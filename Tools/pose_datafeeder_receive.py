import argparse
import json
import time
from LayerApplication.utils.Mqtt import MqttClient


def main():
    parser = argparse.ArgumentParser(description="Receive pose data over MQTT")
    parser.add_argument("--device", required=True, help="device name or id")
    parser.add_argument("--broker", default="localhost", help="MQTT broker address")
    parser.add_argument("--port", type=int, default=1885, help="MQTT broker port")
    args = parser.parse_args()

    mqtt_client = MqttClient(args.broker, args.port)
    topic = f"/DATA/{args.device}/LayerSensing/Pose"
    mqtt_client.mqttc.subscribe(topic)

    def on_message(client, userdata, msg):
        try:
            print(json.loads(msg.payload))
        except json.JSONDecodeError:
            print(msg.payload)

    mqtt_client.mqttc.on_message = on_message
    print(f"Subscribed to {topic}. Waiting for messages...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    mqtt_client.stop()


if __name__ == "__main__":
    main()
