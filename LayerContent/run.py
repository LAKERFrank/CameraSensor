import argparse
import subprocess
import sys

def run(cmd, desc):
    """啟動子行程並回傳 Popen 物件"""
    print(f"▶️  {desc} ...")
    return subprocess.Popen(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run Model3D_mqtt.py and runSensing.py with shared parameters.")
    parser.add_argument("--date", default="2025-04-24_16-33-50", help="Date string, e.g., 2025-04-24_16-33-50")
    parser.add_argument("--camera_idxs", nargs='+', default=["0", "1", "2", "3"], help="List of camera indexes, e.g., 0 1 2 3")
    parser.add_argument("--camera_device", nargs='+', default=[
        "CameraReader_0", "CameraReader_1", "CameraReader_2", "CameraReader_3"
    ], help="List of camera device names, e.g., CameraReader_0 CameraReader_1")
    parser.add_argument("--fps", type=int, default=120, help="FPS value, default = 120")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after processing")
    parser.add_argument("-q", "--quiet", dest="verbose", action="store_false", help="Disable verbose output")

    args = parser.parse_args()

    cmd_model3d = [
        "python3", "Model3D_mqtt.py",
        "--date", args.date,
        "--camera_idxs", *args.camera_idxs,
        "--camera_device", *args.camera_device,
        "--fps", str(args.fps)
    ]

    cmd_tracknet = [
        "python3", "runSensing.py",
        "--date", args.date,
        "--camera_idxs", *args.camera_idxs,
        "--camera_device", *args.camera_device
    ]

    cmd_eval = [
        "python3", "eval/event/evaluation_two_stage_multi.py",
        "--df", f"{args.date}:-",
        "-v"
    ]
    if not args.verbose:
        cmd_eval.remove("-v")

    p_model3d  = run(cmd_model3d,  "Running Model3D_mqtt.py")
    p_tracknet = run(cmd_tracknet, "Running runSensing.py")

    model3d_ret = p_model3d.wait()
    tracknet_ret = p_tracknet.wait()

    if tracknet_ret != 0 or model3d_ret != 0:
        print("❌ One or both processes failed.")
        if model3d_ret != 0:
            print(f"Model3D_mqtt.py exited with code {model3d_ret}")
        if tracknet_ret != 0:
            print(f"runSensing.py exited with code {tracknet_ret}")
        # sys.exit(1)

    if args.eval:
        eval_ret = subprocess.run(cmd_eval).returncode
        if eval_ret != 0:
            print(f"⚠️  evaluation_two_stage.py exited with code {eval_ret}",
                file=sys.stderr)
            sys.exit(eval_ret)

    print("✅ All processes finished successfully.")

if __name__ == "__main__":
    main()
