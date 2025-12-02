import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from LayerApplication.utils.Mqtt import MqttClient
from LayerCamera.camera.RpcCamera import RpcCamera
from LayerSensing.RpcSensing import RpcSensing
from lib.common import ROOTDIR, loadConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TrackNet/Pose and visualize outputs")
    parser.add_argument(
        "--replay-dirname",
        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        help="Replay directory name under <ROOTDIR>/replay",
    )
    parser.add_argument("--cam-idx", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Skip inference and only visualize existing outputs",
    )
    parser.add_argument(
        "--skip-visualize", action="store_true", help="Do not run visualization after inference"
    )
    parser.add_argument(
        "--land-frame",
        type=int,
        help="Landing frame number (1-based). If omitted, it will be inferred from TrackNet CSV.",
    )
    parser.add_argument(
        "--pose-log",
        type=str,
        help="Optional pose results file (json/jsonl/ndjson). If not provided, pose MQTT messages are recorded.",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Optional video path for visualization (defaults to replay/CameraReader_<cam>.mp4)",
    )
    parser.add_argument(
        "--tracknet-csv",
        type=str,
        help="Optional TrackNet CSV path (defaults to replay/TrackNet_<cam>.csv)",
    )
    return parser.parse_args()


def _ensure_tools_on_path() -> None:
    tools_dir = Path(__file__).resolve().parent / "Tools"
    tools_path = str(tools_dir)
    if tools_path not in sys.path:
        sys.path.append(tools_path)


def _subscribe_pose_logs(mqtt_client, pose_entries: List[dict]) -> None:
    pose_topic = "/DATA/+/SensingLayer/Pose"

    def _on_pose(client, _userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return
        if isinstance(data, dict):
            pose_entries.append(data)

    mqtt_client.message_callback_add(pose_topic, _on_pose)
    mqtt_client.subscribe(pose_topic)


def _write_pose_log(entries: List[dict], pose_log_path: Path) -> Optional[Path]:
    if not entries:
        return None
    pose_log_path.parent.mkdir(parents=True, exist_ok=True)
    with pose_log_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return pose_log_path


def _infer_land_frame_from_csv(csv_path: Path) -> Optional[int]:
    if not csv_path.exists():
        return None
    last_visible: Optional[int] = None
    with csv_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            try:
                if int(row.get("Visibility", 0)) == 1:
                    last_visible = idx
            except Exception:
                continue
    return None if last_visible is None else last_visible + 1


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = _parse_args()

    cfg = loadConfig(f"{ROOTDIR}/config")
    mqtt = MqttClient(cfg["Project"]["mqtt_broker"], 1883)

    camera = RpcCamera(f"CameraReader_{args.cam_idx}", mqtt.mqttc)
    sensing = RpcSensing(f"CameraReader_{args.cam_idx}", mqtt.mqttc)

    replay_dir = Path(ROOTDIR) / "replay" / args.replay_dirname
    video_path = Path(args.video) if args.video else replay_dir / f"CameraReader_{args.cam_idx}.mp4"
    tracknet_csv_path = (
        Path(args.tracknet_csv)
        if args.tracknet_csv
        else replay_dir / f"TrackNet_{args.cam_idx}.csv"
    )

    pose_entries: List[dict] = []
    pose_log_path = Path(args.pose_log) if args.pose_log else replay_dir / f"Pose_{args.cam_idx}.jsonl"

    if not args.visualize_only:
        replay_dir.mkdir(parents=True, exist_ok=True)
        _subscribe_pose_logs(mqtt.mqttc, pose_entries)

        ret = sensing.startTrackNet((640, 480), "tracknet_v2", "no114_30.tar", args.replay_dirname, args.cam_idx)
        print(f"TrackNet: {ret}")

        pose_ret = sensing.startPose("pose.engine", args.cam_idx)
        print(f"Pose: {pose_ret}")

        duration = camera.startVideoFeeder(str(video_path))
        time.sleep(duration)

        camera.stopVideoFeeder()
        sensing.stopTrackNet()
        sensing.stopPose()

        saved_pose_path = _write_pose_log(pose_entries, pose_log_path)
        if saved_pose_path:
            print(f"Pose log saved to {saved_pose_path}")
    else:
        saved_pose_path = pose_log_path if pose_log_path.exists() else None

    if args.skip_visualize:
        return

    land_frame = args.land_frame or _infer_land_frame_from_csv(tracknet_csv_path)
    if land_frame is None:
        print("No landing frame found; visualization will omit landing highlight.")

    _ensure_tools_on_path()
    import tracknet_pose_visualize as visualizer

    pose_path = saved_pose_path if saved_pose_path and saved_pose_path.exists() else None
    visualizer.visualize(
        str(video_path),
        str(tracknet_csv_path),
        land_frame=land_frame,
        pose_path=str(pose_path) if pose_path else None,
    )


if __name__ == "__main__":
    main()
