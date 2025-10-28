import os
import subprocess
import re
import argparse
import csv


DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ROOTDIR = os.path.dirname(ROOTDIR)
# Base directory configuration
replay_dir = f"{ROOTDIR}/replay"

def is_target_date(folder_name, date_prefixes):
    # If date_prefixes is empty, process all folders; otherwise, check if the folder name starts with any of the specified prefixes.
    if not date_prefixes:
        return True
    return any(folder_name.startswith(prefix) for prefix in date_prefixes)

def find_missing_dates(date_prefixes):
    missing_dates = []
    speed_dates = []

    # Iterate through folders
    for folder_name in os.listdir(replay_dir):
        folder_path = os.path.join(replay_dir, folder_name)
        smash_info_path = os.path.join(folder_path, "Model3D_smash_info.csv")

        if os.path.isdir(folder_path) and is_target_date(folder_name, date_prefixes):
            if not os.path.exists(smash_info_path):
                missing_dates.append((folder_name, "Model3D_smash_info.csv is missing"))
            else:
                with open(smash_info_path, "r") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row.get("Hit") == "False":
                            missing_dates.append((folder_name, "Hit column is False"))
                            break
                        else:
                            speed_dates.append((folder_name, row.get("Speed")))


    print()
    print('------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------')
    print("SPEED:")
    for date, speed in speed_dates:
        print(f"{date}: {speed}")
    print("The following dates are Missing Model3D_smash_info.csv or the 'Hit' is False:")
    for date, reason in missing_dates:
        print(f"{date}: {reason}")

def process_dates(date_prefixes):
    # Ensure the directory exists
    if not os.path.exists(replay_dir):
        print(f"Directory {replay_dir} does not exist!")
        exit(1)

     # Iterate through folders
    for folder_name in os.listdir(replay_dir):
        folder_path = os.path.join(replay_dir, folder_name)

        # Ensure it's a directory and the name matches the target date format
        if os.path.isdir(folder_path) and is_target_date(folder_name, date_prefixes):
            config_path = os.path.join(folder_path, "config")
            output_csv = os.path.join(folder_path, "Model3D.csv")

            # Construct the command
            command = [
                "python3", f"{ROOTDIR}/LayerContent/CES/Model3D_offline.py",
                "--config", config_path,
                "--output_csv", output_csv,
                "--atleast1hit"
            ]

            print()
            print('------------------------------------------------------------------------------')
            try:
                print(f"執行命令: {' '.join(command)}")
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"執行失敗: {e}")
            except Exception as e:
                print(f"發生錯誤: {e}")

def main(date_prefixes, findmiss, skip_model3d, delete):
    if not skip_model3d:
        # Execute Model3D_offline
        process_dates(date_prefixes)

    # If required, execute the missing check
    if findmiss:
        find_missing_dates(date_prefixes)

    if delete:
        for folder_name in os.listdir(replay_dir):
            folder_path = os.path.join(replay_dir, folder_name)
            if os.path.isdir(folder_path) and is_target_date(folder_name, date_prefixes):
                trajectory_path = os.path.join(folder_path, "trajectory_gradient.png")
                if os.path.exists(trajectory_path):
                    os.remove(trajectory_path)
                    print(f"已刪除: {trajectory_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Model3D_offline.py for specified date folders.")
    parser.add_argument(
        "--dates",
        nargs="*",
        default=[],
        help="Specify the date prefixes to process (e.g., 2024-12-18_). Leave empty to process all folders."
    )
    parser.add_argument(
        "--findmiss",
        action="store_true",
        help="Find folders missing Model3D_smash_info.csv or where the 'Hit' column is False."
    )
    parser.add_argument(
        "--skip_model3d",
        action="store_true",
        help="Skip executing Model3D_offline and only perform the missing check."
    )
    parser.add_argument(
        "--delete",
        action="store_true"
    )
    args = parser.parse_args()
    main(args.dates, args.findmiss, args.skip_model3d, args.delete)
