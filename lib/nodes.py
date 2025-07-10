'''
functions: node related functions
'''
import os
import logging

from PyQt5.QtCore import QProcess

from enum import Enum, auto

from lib.common import ROOTDIR

# setup Camera
def setupCameras(cfg):
    cameras = []
    for node_name, node_info in cfg.items():
        # Get All Cameras
        if 'node_type' in node_info:
            if node_info['node_type'] == 'Reader':
                camera = CameraReader(node_name, node_info)
                cameras.append(camera)
    return cameras

# setup Offline track 3d trajectory
def setupOfflineTrackingNodes(project_name, cfg, replay_path, weights, page=None):
    nodes = []
    for node_name, node_info in cfg.items():
        if 'node_type' in node_info:
            if node_info['node_type'] == 'TrackNet':
                tracknet = TrackNet(project_name, node_name, node_info, replay_path, weights, page)
                nodes.append(tracknet)
    return nodes

class Node():
    class State(Enum):
        NO_START = auto()
        READY = auto()
        TERMINATED = auto()

    def __init__(self):
        self.name = "None"
        self.command = ""
        self.process = None
        self.state = Node.State.NO_START

    def stop(self):
        if self.process is not None:
            if self.process.state() == QProcess.Running:
                self.process.kill()
                self.process.terminate()

    def start(self):
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        logging.debug(self.command)
        self.process.start("/bin/bash", ['-c', f"cd {ROOTDIR} && {self.command}"])

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        logging.debug(stdout)

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        logging.error(stderr)

class CameraReader(Node):
    def __init__(self, node_name, node_info):
        super().__init__()
        self.name = node_name
        self.node_type = node_info['node_type']
        self.brand = node_info['brand']
        self.hw_id = node_info['hw_id']
        self.output_topic = node_info['output_topic']
        self.command = f"python3 -m Reader.{node_info['brand']}.main " f"--nodename {node_name}"
        self.isStreaming = False
        self.gain = 135
        self.camerasensor_hostname = node_info['camerasensor_hostname']

class TrackNet(Node):
    def __init__(self, project_name, node_name, node_info, load_path, weights, page):
        super().__init__()
        self.name = node_name
        self.node_type = node_info['node_type']
        video_path = os.path.realpath(os.path.join(load_path, f"{node_info['file_name']}.mp4"))
        self.command = f"python3 {ROOTDIR}/LayerSensing/TrackNet/TrackNet10/TrackNet.py " f"--nodename {node_name} --weights {weights} --data {video_path} --save_csv {load_path} --page {page}"

class TrackNet_offline(Node):
    def __init__(self, node_name, data, save_csv, weights):
        super().__init__()
        self.name = node_name
        self.command = f"python3 {ROOTDIR}/LayerSensing/TrackNet/TrackNet10/TrackNet.py " f"--nodename {node_name} --weights {weights} --data {data} --save_csv {save_csv}"