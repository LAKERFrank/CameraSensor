from multiprocessing import current_process
import sys
import os
import numpy as np
import copy
import csv
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import argparse
from matplotlib import cm

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ROOTDIR = os.path.dirname(ROOTDIR)
REPLAYDIR = f"{ROOTDIR}/replay/"
print(REPLAYDIR)

sys.path.insert(0, ROOTDIR)
sys.path.insert(0, os.path.join(ROOTDIR, 'lib'))
from lib.point import Point as P
from lib.writer import CSVWriter

def drawTrajectory(date, points=1):
    file = 'Model3D_event_' + str(points) + '.csv'
    file_path = os.path.join(REPLAYDIR, date, file)
    output_file = 'trajectory.png'
    output_path = os.path.join(REPLAYDIR, date, output_file)

    data = pd.read_csv(file_path)
    points = data[data['Visibility'] == 1]
    event_points = points[points['Event'] != 0]
    plt.style.use('default')  # Use the default style with a white background
    plt.figure(figsize=(8, 6))
    plt.gcf().set_facecolor('white')  # Ensure the background is white
    if len(event_points) == 3 and event_points.iloc[0]['Event'] == 2 and event_points.iloc[1]['Event'] == 1 and event_points.iloc[2]['Event'] == 3:
        # serve = points[points['Event'] == 2].index[0]
        serve = event_points.iloc[0].name
        hit = event_points.iloc[1].name
        dead = event_points.iloc[2].name

        segment_serve_to_hit = points.loc[serve:hit]
        plt.scatter(segment_serve_to_hit['Y'], segment_serve_to_hit['Z'], color='gray', label='Serve to Hit', alpha=0.7)
        segment_hit_to_dead = points.loc[hit:dead]
        plt.scatter(segment_hit_to_dead['Y'], segment_hit_to_dead['Z'], color='black', label='Hit to Dead', alpha=0.7)

        for event, color, event_name in zip([2, 1, 3], ['red', 'blue', 'green'], ['Serve', 'Hit', 'Dead']):
            event_data = points[points['Event'] == event]
            plt.scatter(event_data['Y'], event_data['Z'], color=color, label=event_name, s=80)
    else:
        for event, color, event_name in zip([2, 1, 3], ['red', 'blue', 'green'], ['Serve', 'Hit', 'Dead']):
            event_data = points[points['Event'] == event]
            plt.scatter(event_data['Y'], event_data['Z'], color=color, label=event_name, s=80)
        plt.scatter(points['Y'], points['Z'], color='black', alpha=0.5)

    plt.xlim(-7, 3)
    plt.ylim(3, 0)
    plt.gca().invert_yaxis()
    
    plt.title('Y - Z Coordinates of Points')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Coordinate')
    plt.grid(True)
    plt.legend(fontsize=6, loc='lower right')

    plt.savefig(output_path)
    plt.close()

def drawTrajectory_points(save_path, points):
    output_file = 'trajectory.png'
    output_path = os.path.join(save_path, output_file)

    visible_points = [p for p in points if p.visibility == 1]
    event_points = [p for p in visible_points if p.event != 0]
    plt.style.use('default')  # Use the default style with a white background
    plt.figure(figsize=(8, 6))
    plt.gcf().set_facecolor('white')  # Ensure the background is white
    if len(event_points) == 3 and \
       event_points[0].event == 2 and \
       event_points[1].event == 1 and \
       event_points[2].event == 3:
        serve_idx = visible_points.index(event_points[0])
        hit_idx = visible_points.index(event_points[1])
        dead_idx = visible_points.index(event_points[2])

        segment_serve_to_hit = visible_points[serve_idx:hit_idx + 1]
        plt.scatter([p.y for p in segment_serve_to_hit], [p.z for p in segment_serve_to_hit],
                    color='gray', label='Serve to Hit', alpha=0.7)

        segment_hit_to_dead = visible_points[hit_idx:dead_idx + 1]
        plt.scatter([p.y for p in segment_hit_to_dead], [p.z for p in segment_hit_to_dead],
                    color='black', label='Hit to Dead', alpha=0.7)

        for event, color, event_name in zip([2, 1, 3], ['red', 'blue', 'green'], ['Serve', 'Hit', 'Dead']):
            event_data = [p for p in visible_points if p.event == event]
            plt.scatter([p.y for p in event_data], [p.z for p in event_data],
                        color=color, label=event_name, s=80)
    else:
        for event, color, event_name in zip([2, 1, 3], ['red', 'blue', 'green'], ['Serve', 'Hit', 'Dead']):
            event_data = [p for p in visible_points if p.event == event]
            plt.scatter([p.y for p in event_data], [p.z for p in event_data],
                        color=color, label=event_name, s=80)
        plt.scatter([p.y for p in visible_points], [p.z for p in visible_points],
                    color='black', alpha=0.5)

    plt.xlim(-7, 3)
    plt.ylim(3, 0)
    plt.gca().invert_yaxis()

    plt.title('Y - Z Coordinates of Points')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Coordinate')
    plt.grid(True)
    plt.legend(fontsize=6, loc='lower right')

    plt.savefig(output_path)
    plt.close()

def drawGradientTrajectory(date):
    file = 'Model3D_smooth_gradient.csv'
    file_path = os.path.join(REPLAYDIR, date, file)
    output_file = 'trajectory_gradient.png'
    output_path = os.path.join(REPLAYDIR, date, output_file)

    data = pd.read_csv(file_path)
    points = data[data['Visibility'] == 1]
    event_points = points[points['Event'] != 0]
    plt.style.use('default')  # Use the default style with a white background
    plt.figure(figsize=(8, 6))
    plt.gcf().set_facecolor('white')  # Ensure the background is white
    if event_points.iloc[0]['Event'] == 2 and event_points.iloc[1]['Event'] == 1 and event_points.iloc[2]['Event'] == 3:
        # serve = points[points['Event'] == 2].index[0]
        serve = event_points.iloc[0].name
        hit = event_points.iloc[1].name
        dead = event_points.iloc[2].name

        segment_serve_to_hit = points.loc[serve:hit]
        plt.scatter(segment_serve_to_hit['Y'], segment_serve_to_hit['Z'], color='gray', label='Serve to Hit', alpha=0.7)
        segment_hit_to_dead = points.loc[hit:dead]
        plt.scatter(segment_hit_to_dead['Y'], segment_hit_to_dead['Z'], color='pink', label='Hit to Dead', alpha=0.5, s=5)

        for event, color, event_name in zip([2, 1, 3], ['red', 'blue', 'green'], ['Serve', 'Hit', 'Dead']):
            event_data = points[points['Event'] == event]
            plt.scatter(event_data['Y'], event_data['Z'], color=color, label=event_name, s=80)
    else:
        for event, color, event_name in zip([2, 1, 3], ['red', 'blue', 'green'], ['Serve', 'Hit', 'Dead']):
            event_data = points[points['Event'] == event]
            plt.scatter(event_data['Y'], event_data['Z'], color=color, label=event_name, s=80)
        plt.scatter(points['Y'], points['Z'], color='black', alpha=0.5)

    plt.xlim(-7, 3)
    plt.ylim(3, 0)
    plt.gca().invert_yaxis()
    
    plt.title('Y - Z Coordinates of Points')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Coordinate')
    plt.grid(True)
    plt.legend(fontsize=6, loc='lower right')

    plt.savefig(output_path)
    plt.close()

def drawGradientTrajectory_points(save_path, points):
    output_file = 'trajectory_gradient.png'
    output_path = os.path.join(save_path, output_file)

    visible_points = [p for p in points if p.visibility == 1]
    event_points = [p for p in visible_points if p.event != 0]
    plt.style.use('default')  # Use the default style with a white background
    plt.figure(figsize=(8, 6))
    plt.gcf().set_facecolor('white')  # Ensure the background is white
    if len(event_points) >= 3 and \
       event_points[0].event == 2 and \
       event_points[1].event == 1 and \
       event_points[2].event == 3:
        serve_idx = visible_points.index(event_points[0])
        hit_idx = visible_points.index(event_points[1])
        dead_idx = visible_points.index(event_points[2])

        segment_serve_to_hit = visible_points[serve_idx:hit_idx + 1]
        plt.scatter([p.y for p in segment_serve_to_hit], [p.z for p in segment_serve_to_hit],
                    color='gray', label='Serve to Hit', alpha=0.7)

        segment_hit_to_dead = visible_points[hit_idx:dead_idx + 1]
        plt.scatter([p.y for p in segment_hit_to_dead], [p.z for p in segment_hit_to_dead],
                    color='pink', label='Hit to Dead', alpha=0.5, s=5)

        for event, color, event_name in zip([2, 1, 3], ['red', 'blue', 'green'], ['Serve', 'Hit', 'Dead']):
            event_data = [p for p in visible_points if p.event == event]
            plt.scatter([p.y for p in event_data], [p.z for p in event_data],
                        color=color, label=event_name, s=80)
    else:
        for event, color, event_name in zip([2, 1, 3], ['red', 'blue', 'green'], ['Serve', 'Hit', 'Dead']):
            event_data = [p for p in visible_points if p.event == event]
            plt.scatter([p.y for p in event_data], [p.z for p in event_data],
                        color=color, label=event_name, s=80)
        plt.scatter([p.y for p in visible_points], [p.z for p in visible_points],
                    color='black', alpha=0.5)

    plt.xlim(-7, 3)
    plt.ylim(3, 0)
    plt.gca().invert_yaxis()

    plt.title('Y - Z Coordinates of Points')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Coordinate')
    plt.grid(True)
    plt.legend(fontsize=6, loc='lower right')

    plt.savefig(output_path)
    plt.close()

def drawVisualization(date):
    output_path = os.path.join(REPLAYDIR, date, 'visualization.png')
    images = []
    for img in ['TrackNet_0.png', 'TrackNet_1.png', 'trajectory.png', 'trajectory_gradient.png']:
        img_path = os.path.join(REPLAYDIR, date, img)
        if os.path.exists(img_path):
            images.append(Image.open(img_path))
        else:
            images.append(None)
    max_width = max(img.size[0] if img else 0 for img in images)
    max_height = max(img.size[1] if img else 0 for img in images)
    total_width = 2 * max_width
    total_height = 2 * max_height
    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255, 0))
    positions = [
        (0, 0),
        (max_width, 0),
        (0, max_height),
        (max_width, max_height)
    ]
    for img, pos in zip(images, positions):
        if img:
            combined_image.paste(img, pos)
    combined_image.save(output_path)

def drawPoints(points, save_path='.'):
    x_coords = [point.x for point in points]
    y_coords = [point.y for point in points]
    z_coords = [point.z for point in points]
    fid = [point.fid for point in points]

    plt.style.use('default')  # Use the default style with a white background
    plt.figure(figsize=(8, 6))
    plt.gcf().set_facecolor('white')  # Ensure the background is white
    plt.xlim(-7, 7)
    plt.ylim(7, 0)
    plt.gca().invert_yaxis()

    norm = plt.Normalize(min(fid), max(fid))
    cmap = cm.get_cmap('plasma')
    colors = cmap(norm(fid))
    plt.scatter(y_coords, z_coords, color=colors, s=5, label='Points')
    # plt.plot(y_coords, z_coords, color='red', linestyle='--', label='Path')

    # plt.plot(y_coords, color='red', linestyle='--', label='Path')
    
    plt.title('Y - Z Coordinates of Points')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Coordinate')
    plt.grid(True)
    plt.legend(fontsize=6, loc='lower right')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f'{save_path}/points.png')
    plt.close()


def is_target_date(folder_name, date_prefixes):
    # If date_prefixes is empty, process all folders; otherwise, check if the folder name starts with any of the specified prefixes.
    if not date_prefixes:
        return True
    return any(folder_name.startswith(prefix) for prefix in date_prefixes)

def process_dates(date_prefixes):
    for folder_name in os.listdir(REPLAYDIR):
        folder_path = os.path.join(REPLAYDIR, folder_name)

        # Ensure it's a directory and the name matches the target date format
        if os.path.isdir(folder_path) and is_target_date(folder_name, date_prefixes):
            print(folder_path)
            drawVisualization(folder_path)

def main(date_prefixes):
    process_dates(date_prefixes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Model3D_offline.py for specified date folders.")
    parser.add_argument(
        "--dates",
        nargs="*",
        default=[],
        help="Specify the date prefixes to process (e.g., 2024-12-18_). Leave empty to process all folders."
    )
    args = parser.parse_args()
    main(args.dates)



