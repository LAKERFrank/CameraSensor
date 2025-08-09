import argparse
import paho.mqtt.client as mqtt
from LayerSensing.PoseDatafeeder import PoseDatafeeder


def main():
    parser = argparse.ArgumentParser(description="Send pose data via PoseDatafeeder")
    parser.add_argument("--broker", default="localhost", help="MQTT broker address")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--device", required=True, help="Device name used in MQTT topics")
    parser.add_argument("--csv", required=True, help="Pose CSV file path")
    parser.add_argument("--meta", help="Optional meta CSV file path")
    args = parser.parse_args()

    mqttc = mqtt.Client()
    mqttc.connect(args.broker, args.port)
    mqttc.loop_start()

    feeder = PoseDatafeeder(mqttc, args.device, args.csv, args.meta)
    feeder.start()
    feeder.join()

    mqttc.loop_stop()
    mqttc.disconnect()


if __name__ == "__main__":
    main()
