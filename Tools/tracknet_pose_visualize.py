"""Visualize TrackNet ball positions and pose keypoints on a video.

The script overlays TrackNet CSV outputs (red dots) and pose detections
(blue keypoints + skeleton) onto a provided video and saves a new video
next to the inputs. It can also be imported as a module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import pandas as pd

# Pose skeleton (COCO/YOLOv8 order with 17 keypoints)
POSE_SKELETON: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

RED = (0, 0, 255)
BLUE = (255, 0, 0)


def _load_tracknet_points(csv_path: Path) -> Dict[int, Tuple[float, float]]:
    df = pd.read_csv(csv_path)

    # Prefer the Visibility flag when present but do not drop rows if the column
    # is missing or all visibilities are zero (some exports never mark the ball
    # as visible). This keeps a point available for every frame TrackNet
    # produced.
    if "Visibility" in df.columns and df["Visibility"].sum() > 0:
        df = df[df["Visibility"] == 1]

    # Normalize frame index to zero-based to match OpenCV's frame counter.
    if df["Frame"].min() >= 1 and 0 not in set(df["Frame"].tolist()):
        df = df.assign(Frame=df["Frame"] - 1)

    # Ensure the latest point per frame is used if duplicates exist.
    return {int(row.Frame): (float(row.X), float(row.Y)) for row in df.itertuples(index=False)}


def _load_pose_entries(pose_path: Path) -> Dict[int, List[List[List[float]]]]:
    """Load pose detections keyed by frame index.

    Each line in the pose file should be a JSON object published by the
    pose worker with a ``frame_index`` and ``detections`` list. Missing or
    malformed lines are skipped to keep visualization resilient.
    """

    frames: Dict[int, List[List[List[float]]]] = {}
    for line in pose_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        frame_idx_raw = payload.get("frame_index")
        detections = payload.get("detections")
        try:
            frame_idx = int(frame_idx_raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(detections, list):
            continue
        keypoints_list = []
        for det in detections:
            kpts = det.get("keypoints") if isinstance(det, dict) else None
            if isinstance(kpts, list):
                keypoints_list.append(kpts)
        if keypoints_list:
            frames.setdefault(frame_idx, []).extend(keypoints_list)

    if frames and min(frames) >= 1 and 0 not in frames:
        frames = {idx - 1: detections for idx, detections in frames.items()}
    return frames


def _draw_pose(image, keypoints: Iterable[List[float]]) -> None:
    points = []
    for kp in keypoints:
        if len(kp) < 2:
            points.append(None)
            continue
        x, y = kp[0], kp[1]
        conf_ok = len(kp) < 3 or kp[2] > 0
        if conf_ok:
            cv2.circle(image, (int(x), int(y)), 3, BLUE, -1, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else:
            points.append(None)
    for a, b in POSE_SKELETON:
        if a < len(points) and b < len(points):
            pa, pb = points[a], points[b]
            if pa and pb:
                cv2.line(image, pa, pb, BLUE, 2, lineType=cv2.LINE_AA)


def visualize(
    video_path: Path,
    tracknet_csv: Path,
    *,
    pose_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    video_path = video_path.resolve()
    tracknet_csv = tracknet_csv.resolve()
    pose_path = pose_path.resolve() if pose_path is not None else None

    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_tracknet_pose.mp4")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not tracknet_csv.exists():
        raise FileNotFoundError(f"TrackNet CSV not found: {tracknet_csv}")

    tracknet_points = _load_tracknet_points(tracknet_csv)
    pose_entries = _load_pose_entries(pose_path) if pose_path and pose_path.exists() else {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_idx in tracknet_points:
            x, y = tracknet_points[frame_idx]
            cv2.circle(frame, (int(x), int(y)), 6, RED, -1, lineType=cv2.LINE_AA)

        for detection in pose_entries.get(frame_idx, []):
            _draw_pose(frame, detection)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="TrackNet + Pose visualization")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--csv", required=True, help="TrackNet CSV path")
    parser.add_argument("--pose", help="Pose JSONL/NDJSON path")
    parser.add_argument(
        "--output",
        help="Output video path (default: <video>_tracknet_pose.mp4)",
    )
    args = parser.parse_args()

    output = visualize(
        Path(args.video),
        Path(args.csv),
        pose_path=Path(args.pose) if args.pose else None,
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Visualization saved to: {output}")


if __name__ == "__main__":
    main()
