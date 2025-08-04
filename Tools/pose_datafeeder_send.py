import argparse
from LayerApplication.utils.Mqtt import MqttClient
from LayerSensing.PoseManager import PoseManager
from LayerCamera.CameraSystemC.recorder_module import ImageBuffer


def main():
    parser = argparse.ArgumentParser(description="Send pose CSV data over MQTT")
    parser.add_argument("--device", required=True, help="device name or id")
    parser.add_argument("--csv", required=True, help="pose CSV file path")
    parser.add_argument("--meta", help="optional meta CSV file for timestamps")
    parser.add_argument("--broker", default="localhost", help="MQTT broker address")
    parser.add_argument("--port", type=int, default=1885, help="MQTT broker port")
    args = parser.parse_args()

    mqtt_client = MqttClient(args.broker, args.port)
    pose_manager = PoseManager(str(args.device), mqtt_client.mqttc, ImageBuffer())
    pose_manager.startDatafeeder(args.csv, args.meta)
    pose_manager.stopDatafeeder()
    mqtt_client.stop()


if __name__ == "__main__":
    main()
