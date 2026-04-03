#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

import paho.mqtt.client as mqtt


@dataclass
class StreamStats:
    pose_msgs: int = 0
    pose_frames: List[int] = field(default_factory=list)
    pose_empty_det_msgs: int = 0
    pose_nonempty_det_msgs: int = 0
    pose_eof_msgs: int = 0

    tracknet_msgs: int = 0
    tracknet_frames: List[int] = field(default_factory=list)
    tracknet_nonempty_linear_msgs: int = 0
    tracknet_empty_linear_msgs: int = 0
    tracknet_eof_msgs: int = 0

    parse_errors: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Pose and TrackNet both consume and publish frames")
    parser.add_argument("--broker", default="127.0.0.1", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--device", default="test-0", help="Device name used by sensing layer")
    parser.add_argument("--duration", type=float, default=15.0, help="Seconds to monitor topics")
    parser.add_argument("--min-pose-msgs", type=int, default=3)
    parser.add_argument("--min-tracknet-msgs", type=int, default=1)
    return parser.parse_args()


def decode_json(payload: bytes) -> Optional[dict]:
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    stats = StreamStats()

    pose_topic = f"/DATA/{args.device}/SensingLayer/Pose"
    tracknet_topic = f"/DATA/{args.device}/SensingLayer/TrackNet"

    def on_connect(client, userdata, flags, reason_code, properties=None):
        print(f"[info] connected: reason_code={reason_code}")
        client.subscribe(pose_topic)
        client.subscribe(tracknet_topic)
        print(f"[info] subscribed: {pose_topic}")
        print(f"[info] subscribed: {tracknet_topic}")

    def on_message(client, userdata, msg):
        data = decode_json(msg.payload)
        if data is None:
            stats.parse_errors += 1
            return

        if msg.topic.endswith("/Pose"):
            stats.pose_msgs += 1
            if "frame_id" in data:
                stats.pose_frames.append(int(data["frame_id"]))
            detections = data.get("detection", [])
            if len(detections) == 0:
                stats.pose_empty_det_msgs += 1
            else:
                stats.pose_nonempty_det_msgs += 1
            if data.get("EOF") is True:
                stats.pose_eof_msgs += 1

        elif msg.topic.endswith("/TrackNet"):
            stats.tracknet_msgs += 1
            linear = data.get("linear", [])
            if len(linear) == 0:
                stats.tracknet_empty_linear_msgs += 1
            else:
                stats.tracknet_nonempty_linear_msgs += 1
                for p in linear:
                    fid = p.get("id")
                    if fid is not None:
                        stats.tracknet_frames.append(int(fid))
            if data.get("EOF") is True:
                stats.tracknet_eof_msgs += 1

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[info] connecting to mqtt://{args.broker}:{args.port}")
    client.connect(args.broker, args.port, keepalive=60)
    client.loop_start()

    started = time.time()
    try:
        while time.time() - started < args.duration:
            time.sleep(0.2)
    finally:
        client.loop_stop()
        client.disconnect()

    print("\n=== stream report ===")
    print(f"pose_msgs={stats.pose_msgs}, pose_frames={len(stats.pose_frames)}, "
          f"pose_empty={stats.pose_empty_det_msgs}, pose_nonempty={stats.pose_nonempty_det_msgs}, pose_eof={stats.pose_eof_msgs}")
    print(f"tracknet_msgs={stats.tracknet_msgs}, tracknet_frames={len(stats.tracknet_frames)}, "
          f"tracknet_empty={stats.tracknet_empty_linear_msgs}, tracknet_nonempty={stats.tracknet_nonempty_linear_msgs}, "
          f"tracknet_eof={stats.tracknet_eof_msgs}")
    print(f"parse_errors={stats.parse_errors}")

    ok = True
    if stats.pose_msgs < args.min_pose_msgs:
        print(f"[fail] pose messages too few: {stats.pose_msgs} < {args.min_pose_msgs}")
        ok = False
    if stats.tracknet_msgs < args.min_tracknet_msgs:
        print(f"[fail] tracknet messages too few: {stats.tracknet_msgs} < {args.min_tracknet_msgs}")
        ok = False

    if stats.pose_msgs > 0 and stats.pose_nonempty_det_msgs == 0:
        print("[warn] pose published only empty detections (likely placeholder inference output).")

    if stats.tracknet_msgs > 0 and stats.tracknet_nonempty_linear_msgs == 0:
        print("[warn] tracknet published only empty linear payloads.")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
