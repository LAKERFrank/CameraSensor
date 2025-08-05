import argparse
import logging

from LayerSensing.SensingAgent import SensingLayerAgent
from lib.GracefulKiller import GracefulKiller
from LayerCamera.CameraAgent import CameraLayerAgent
from lib.common import ROOTDIR, loadConfig

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="host name")
    parser.add_argument("serial", help="camera serial")
    parser.add_argument(
        "pose_fps",
        type=int,
        nargs="?",
        default=30,
        help="pose estimation FPS, options: 120, 60, 40, 30, 20",
    )
    args = parser.parse_args()

    # loading project config
    cfg_file = f"{ROOTDIR}/config"
    cfg = loadConfig(cfg_file)
    broker_ip = cfg["Project"]["mqtt_broker"]
    broker_port = int(cfg["Project"]["mqtt_port"])

    device_name = args.name
    pose_fps = args.pose_fps  # supported pose fps: 120, 60, 40, 30, 20

    cameraAgent = CameraLayerAgent(device_name, args.serial)
    cameraAgent.start(broker_ip, broker_port)

    sensingAgent = SensingLayerAgent(device_name, cameraAgent.camera.getImageBuffer(),
                                     pose_fps=pose_fps)
    sensingAgent.start(broker_ip, broker_port)

    logging.info(f"{__file__} started.")

    GracefulKiller().wait()

    cameraAgent.stop()
    sensingAgent.stop()
    logging.info(f"{__file__} stopped.")
