import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import cv2

POSE_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # torso + legs
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video & TrackNet/Pose Visualization Tool")
    parser.add_argument("--video", type=str, required=True, help="video file name")
    parser.add_argument("--csv", type=str, required=True, help="csv(should contain Frame,Visibility,X,Y)")
    parser.add_argument(
        "--land",
        type=int,
        help="Land point frame number (1-based). If omitted, no landing frame is highlighted.",
    )
    parser.add_argument("--pose", type=str, help="Pose results file (json/jsonl/ndjson)")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Optional output directory (defaults to <video_dir>/output)",
    )
    return parser.parse_args()


def load_tracknet_points(csv_path: str) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, float], Dict[int, float]]:
    frame: Dict[int, int] = {}
    visibility: Dict[int, int] = {}
    x: Dict[int, float] = {}
    y: Dict[int, float] = {}
    with open(csv_path, newline="") as csvfile:
        rows = csv.DictReader(csvfile)
        for i, row in enumerate(rows):
            frame[i] = int(row["Frame"])
            visibility[i] = int(row["Visibility"])
            x[i] = float(row["X"])
            y[i] = float(row["Y"])
    return frame, visibility, x, y


def _iter_pose_entries(path: Path) -> Iterable[MutableMapping]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, MutableMapping):
                yield data
            else:
                raise ValueError("Each JSON line must be an object")
        return

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, MutableMapping):
                yield item
            else:
                raise ValueError("Pose entry must be a JSON object")
        return

    if isinstance(raw, MutableMapping):
        for key in ("frames", "detections", "data"):
            values = raw.get(key)
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, MutableMapping):
                        yield item
                    else:
                        raise ValueError("Pose entry must be a JSON object")
                return
    raise ValueError("Unsupported pose data format; provide a JSON array or JSONL file")


def _pose_entry_timestamp(entry: MutableMapping) -> Optional[float]:
    for key in ("monotonic_timestamp", "timestamp"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def load_pose_by_frame(path: Optional[str], fps: float) -> Dict[int, List[Sequence[Sequence[float]]]]:
    if path is None:
        return {}

    pose_path = Path(path)
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)

    entries = list(_iter_pose_entries(pose_path))
    if not entries:
        return {}

    first_ts = _pose_entry_timestamp(entries[0])
    pose_by_frame: Dict[int, List[Sequence[Sequence[float]]]] = {}

    for idx, entry in enumerate(entries):
        frame_idx = entry.get("frame_index")
        if not isinstance(frame_idx, int):
            ts = _pose_entry_timestamp(entry)
            if ts is not None and first_ts is not None and fps > 0:
                frame_idx = int(round((ts - first_ts) * fps))
        if not isinstance(frame_idx, int):
            continue

        detections = entry.get("detections")
        if not isinstance(detections, list):
            continue
        keypoints_list: List[Sequence[Sequence[float]]] = []
        for det in detections:
            if not isinstance(det, MutableMapping):
                continue
            kpts = det.get("keypoints")
            if isinstance(kpts, list):
                keypoints_list.append(kpts)
        if keypoints_list:
            pose_by_frame.setdefault(frame_idx, []).extend(keypoints_list)

    return pose_by_frame


def draw_tracknet_point(image, draw_x: int, draw_y: int, is_land: bool, frame_idx: int):
    color = (0, 255, 0) if is_land else (0, 0, 255)
    cv2.circle(image, (draw_x, draw_y), 8, color, -1)
    cv2.putText(
        image,
        f"{frame_idx}",
        (draw_x, draw_y + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_pose_keypoints(image, poses: List[Sequence[Sequence[float]]]):
    color = (255, 0, 0)  # Blue in BGR
    for keypoints in poses:
        coords: List[Tuple[int, int]] = []
        for kp in keypoints:
            if not isinstance(kp, (list, tuple)) or len(kp) < 2:
                coords.append((-1, -1))
                continue
            score_ok = len(kp) < 3 or kp[2] > 0
            if not score_ok:
                coords.append((-1, -1))
                continue
            coords.append((int(kp[0]), int(kp[1])))

        for pt in coords:
            if pt == (-1, -1):
                continue
            cv2.circle(image, pt, 4, color, -1)

        for a, b in POSE_EDGES:
            if a >= len(coords) or b >= len(coords):
                continue
            if coords[a] == (-1, -1) or coords[b] == (-1, -1):
                continue
            cv2.line(image, coords[a], coords[b], color, 2)


def visualize(
    video_path: str,
    tracknet_csv_path: str,
    *,
    land_frame: Optional[int] = None,
    pose_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Render TrackNet (red) and Pose (blue) results on frames and return the video path."""

    videofile_basename = os.path.splitext(os.path.basename(video_path))[0]
    directory = output_dir or os.path.join(os.path.dirname(video_path), "output")
    os.makedirs(directory, exist_ok=True)

    frame, visibility, x, y = load_tracknet_points(tracknet_csv_path)

    video = cv2.VideoCapture(video_path)
    currentFrame = 0
    fps = float(video.get(cv2.CAP_PROP_FPS))
    output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video_path = os.path.join(directory, videofile_basename + "_visualize.mp4")
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    pose_by_frame = load_pose_by_frame(pose_path, fps)
    landPoint_frame = land_frame - 1 if land_frame is not None else None

    while True:
        success, image = video.read()
        if not success:
            break

        if currentFrame in visibility and visibility[currentFrame] == 1:
            draw_x = int(x[currentFrame])
            draw_y = int(y[currentFrame])
            is_land = landPoint_frame is not None and currentFrame == landPoint_frame
            draw_tracknet_point(image, draw_x, draw_y, is_land, frame[currentFrame])

        if currentFrame in pose_by_frame:
            draw_pose_keypoints(image, pose_by_frame[currentFrame])

        filename = os.path.join(directory, "frame{:0>4d}.jpg".format(currentFrame))
        cv2.imwrite(filename, image)
        print("將 Frame {} 輸出至檔案 {}".format(currentFrame, filename))

        output_video.write(image)
        currentFrame = currentFrame + 1

    video.release()
    output_video.release()
    return output_video_path


def main():
    args = parse_args()
    visualize(
        args.video,
        args.csv,
        land_frame=args.land,
        pose_path=args.pose,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
