import ast
import logging
import numpy as np
import gi
import shutil
import configparser
import time
from datetime import datetime
from pathlib import Path

from LayerCamera.CameraSystemC import recorder_module
from LayerCamera.CameraSystemC.recorder_module import ImageBuffer
from LayerCamera.VideoClipper import VideoClipper
from lib.common import ROOTDIR, loadConfig, saveConfig

gi.require_version("Gst", "1.0")

class CameraConfig():

    PROPERTY_NAMES = [
        "BlackLevel",
        "ExposureAuto",
        "ExposureTime",
        "GainAuto",
        "Gain",
        "BalanceWhiteAuto",
        "BalanceWhiteRed",
        "BalanceWhiteBlue",
        "BalanceWhiteGreen",
    ]

    def open(self, brand, serial):
        self.config_path = Path(ROOTDIR) / "Reader" / brand / "config" / f"{serial}.cfg"
        self.cfg:configparser.ConfigParser = loadConfig(self.config_path)

        # Add default sections
        if not self.cfg.has_section("Camera"):
            self.cfg.add_section("Camera")

        if not self.cfg.has_section("Other"):
            self.cfg.add_section("Other")

    def save(self):
        saveConfig(self.config_path, self.cfg)

    def __getitem__(self, key):
        return self.cfg[key]

    def __setitem__(self, key, value):
        self.cfg[key] = value

class Camera():
    def __init__(self, serial:str, hostname:str=""):
        """
        Args:
            serial (str): serial number of the camera
            hostname (str): hostname of MQTT Camera Agent
        """
        self.recorder = recorder_module.Recorder(ROOTDIR)
        self.brand = "Image_Source"
        self.serial = serial
        self.hostname = hostname

        self.udp_host = None
        self.udp_port = None
        self.preview_window_id = None
        self.direction = 0
        self.software_trigger = False
        self.enable_resync = False
        self.timestamp_ns = 0

        self.width = None
        self.height = None
        self.fps = None
        self.skipping = None

        self.config = CameraConfig()
        self.config.open(self.brand, self.serial)

    @staticmethod
    def getAvailableCameras():
        """list available camera on the host

        Returns:
            dict[str:str]: { serial: cam model name }
        """
        recorder = recorder_module.Recorder(ROOTDIR)
        return recorder.listAvailableCamera()

    def init(self, udp_host:str=None, udp_port:int=None, preview_win_id:int=None, software_trigger:bool=False, resync:bool=False):
        """Initialize Camera

        Args:
            udp_host (str, optional): Network Streaming host (Defaults to None)
            udp_port (int, optional): Network Streaming port (Defaults to None)
            direction (int, optional): Preview Direction. Defaults to 0 (no rotation). value reference to: https://gstreamer.freedesktop.org/documentation/videofilter/videoflip.html?gi-language=c#GstVideoFlipMethod
            preview_win_id (int, optional): Preview Window ID
        """

        if udp_host is not None and udp_host != "":
            self.udp_host = udp_host
        if udp_port is not None and udp_port != "":
            self.udp_port = udp_port
        if preview_win_id is not None and preview_win_id != 0:
            self.preview_window_id = preview_win_id
        if type(software_trigger) is bool:
            self.software_trigger = software_trigger
        if type(resync) is bool:
            self.enable_resync = resync

        if self.isStreaming():
            # already streaming, do nothing
            return

        if (d := self.config["Camera"].get("direction", None)) is not None:
            self.direction = int(d)

        self.recorder.init(
            self.serial,
            0 if self.preview_window_id is None else self.preview_window_id,
            "" if self.udp_host is None else self.udp_host,
            0 if self.udp_port is None else self.udp_port,
            self.direction,
            clockoverlay=False,
            software_trigger=self.software_trigger,
            resync=self.enable_resync
            )

        # initialize resolution/fps/skipping from config
        if (res := self.config["Camera"].get("RecordResolution", None)) is not None \
            and (fps := self.config["Camera"].get("fps", None)) is not None:

            # defaults to 1x1
            skipping = self.config["Camera"].get("skipping", "1x1")

            record_resolution = ast.literal_eval(res)
            self.recorder.setCaptureFormat(*record_resolution, int(fps), skipping)

            self.width, self.height = record_resolution
            self.fps = int(fps)
            self.skipping = skipping

        # initialize properties from config
        for name in CameraConfig.PROPERTY_NAMES:
            if (value := self.config["Camera"].get(name, None)) is not None:
                #print(f"set Property {name} to {value}")
                self.recorder.setProperty(name, str(value))

        return True

    def start(self, timestamp_ns=None):
        if self.isStreaming():
            # already streaming, do nothing
            return
        if timestamp_ns is not None:
            self.timestamp_ns = timestamp_ns
        print(self.hostname, "started:", time.monotonic())
        self.recorder.start(trigger_start=self.timestamp_ns)

    def resync(self, timestamp_ns, dt_ns):
        self.recorder.resync(timestamp_ns, dt_ns)

    def startDisplay(self, win_id:int):
        self.preview_window_id = win_id
        self.recorder.enableDisplay(win_id)

    def stopDisplay(self):
        self.recorder.disableDisplay()

    def startUdp(self, host:str, port:int):
        self.udp_host = host
        self.udp_port = port
        self.recorder.enableUdp(host, port)

    def stopUdp(self):
        self.recorder.disableUdp()

    def release(self):
        """Close Camera
        """
        self.recorder.release()

    def setCameraParameters(self, values:'dict[str, int|float|str]'):
        """Set Camera Properties

        Args:
            values (dict[str, int|float|str]): { Property name : property value }
        """
        for k, v in values.items():
            self.config["Camera"][k] = str(v)
            self.recorder.setProperty(k, str(v))
        self.config.save()

    def getCameraParameters(self):
        """Get Camera Properties

        Returns:
            dict[str, str|float]: { Property name : property value }
        """

        return {
            "BlackLevel"        : float(self.config['Camera'].get('BlackLevel', 0)),
            "ExposureTime"      : float(self.config['Camera'].get('ExposureTime', 0)),
            "Gain"              : float(self.config['Camera'].get('Gain', 0)),
            "BalanceWhiteRed"   : float(self.config['Camera'].get('BalanceWhiteRed', 0)),
            "BalanceWhiteBlue"  : float(self.config['Camera'].get('BalanceWhiteBlue', 0)),
            "BalanceWhiteGreen" : float(self.config['Camera'].get('BalanceWhiteGreen', 0)),
            "direction"         : int(self.config['Camera'].get('direction', 0)),
            "RecordResolution"  : ast.literal_eval(self.config["Camera"].get("RecordResolution", "(0, 0)")),
            "fps"               : str(self.config["Camera"].get("fps", 0)),
            "skipping"          : str(self.config["Camera"].get("skipping", "")),
        }

    def getCaptureFormats(self):
        """Get Camera supported capture formats

        Returns:
            list[dict[str, int | str | list[int]]]:
                [ { width: (int), height: (int), skipping: (str), target_fps: list[int] } ]
        """
        return self.recorder.getCaptureFormats()

    def setCaptureFormat(self, width:int=None, height:int=None, fps:int=None, skipping:str=None, direction:int=None):
        """Set Camera capture format

        Args:
            width (int): width
            height (int): height
            fps (int): fps
            skipping (str): skipping
            direction (int): direction
        """

        resolution_changed = width is not None and height is not None \
                             and self.config["Camera"].get("RecordResolution", "") != str(tuple([width, height]))
        fps_changed = fps is not None and int(self.config["Camera"].get("fps", "")) != fps
        skipping_changed = skipping is not None and self.config["Camera"].get("skipping", "") != skipping
        direction_changed = direction is not None and self.config["Camera"].get("direction", "") != str(direction)

        # check if capture format is changed
        if resolution_changed:
            self.config["Camera"]["RecordResolution"] = str(tuple([width, height]))
        if fps_changed:
            self.config["Camera"]["fps"] = str(fps)
        if direction_changed:
            self.config["Camera"]["direction"] = str(direction)
        if skipping_changed:
            self.config["Camera"]["skipping"] = skipping

        if resolution_changed or fps_changed or skipping_changed or direction_changed:
            self.config.save()

            self.release()
            self.init()
            self.start()

    def getDeviceInfo(self):
        """Get Information about Camera

        Returns:
            dict[str, str]: { brand: "", serial: "", model: "" }
        """
        return self.recorder.getDeviceInfo()

    def getSnapshot(self):
        """Take snapshot
        """
        if snapshot := self.recorder.takeSnapshot():
            return snapshot.image
        return None

    def getImageBuffer(self) -> ImageBuffer:
        return self.recorder.getImageBuffer()

    def startRecording(self, image_buf:bool, image_res:'tuple[int, int]', save_dirname:str, cam_idx:int, mode:str="h264_low"):
        """Start Video Recording

        Args:
            image_buf (bool): Send frames to ImageBuffer or not
            image_res (tuple[int, int]): Image resolution for ImageBuffer
            save_dirname (str): directory name for saving video under PROJECT_ROOT/replay
            cam_idx (int): index of the camera
        """
        save_path = Path(ROOTDIR) / "replay" / save_dirname / f"CameraReader_{cam_idx}.mp4"
        width, height = image_res
        self.recorder.startRecording(str(save_path), image_buf, width, height, mode)
        # copy camera config
        shutil.copy(self.config.config_path, str(save_path)[:-4]+".cfg")

    def stopRecording(self):
        self.recorder.stopRecording()

    #def getFile(self, dirname):
    #    logging.info("get recent recording file")
    #    # get last dir
    #    ret = {}
    #    if (self.save_dir / dirname).exists():
    #        with open((self.save_dir / dirname / "CameraReader.mp4"), 'rb') as file:
    #            ret["video"] = file.read()
    #        with open((self.save_dir / dirname / "CameraReader_meta.csv"), 'rb') as file:
    #            ret["meta"] = file.read()

    #    return ret

    def startVideoFeeder(self, video_path:str, enable_imgbuf:bool=True):
        """Run test video, send frames to Image Buffer
        This will simulate real playing speed.

        Args:
            video_path (str): path of test video
            enable_imgbuf (bool, optional): Send frames to ImageBuffer. Defaults to True.

        Raises:
            Exception: "file not found"

        Returns:
            dict[str, float]: Information of the video. { duration: ... }
        """
        if not Path(video_path).exists():
            raise Exception(f"File {video_path} not found")

        logging.info(f"startVideoFeeder on \"{video_path}\"")

        return self.recorder.startVideoFeeder(video_path, enable_imgbuf)

    def stopVideoFeeder(self):
        return self.recorder.stopVideoFeeder()

    def setIntrinsic(self, ks:np.ndarray, dist:np.ndarray, newcameramtx:np.ndarray):
        """Save camera intrinsic parameters.

        Args:
            ks (np.ndarray): _description_
            dist (np.ndarray): _description_
            newcameramtx (np.ndarray): _description_
        """
        self.config.cfg["Other"]["ks"] = str(ks.tolist())
        self.config.cfg["Other"]["dist"] = str(dist.tolist())
        self.config.cfg["Other"]["newcameramtx"] = str(newcameramtx.tolist())
        self.config.save()

    def getIntrinsic(self):
        """Get camera intrinsic parameters

        Returns:
            dict[str, list]: _description_
        """
        return {
            "ks"            : ast.literal_eval(self.config.cfg["Other"].get("ks", "[]")),
            "dist"          : ast.literal_eval(self.config.cfg["Other"].get("dist", "[]")),
            "newcameramtx"  : ast.literal_eval(self.config.cfg["Other"].get("newcameramtx", "[]")),
        }

    def setExtrinsic(self, poses:np.ndarray, eye:np.ndarray, hmtx:np.ndarray, projection_mat:np.ndarray, extrinsic_mat:np.ndarray):
        """Save camera extrinsic parameters

        Args:
            poses (np.ndarray): _description_
            eye (np.ndarray): _description_
            hmtx (np.ndarray): _description_
            projection_mat (np.ndarray): _description_
            extrinsic_mat (np.ndarray): _description_
        """
        self.config.cfg['Other']['poses'] = str(poses.tolist())
        self.config.cfg['Other']['eye'] = str(eye.tolist())
        self.config.cfg['Other']['hmtx'] = str(hmtx.tolist())
        self.config.cfg['Other']['projection_mat'] = str(projection_mat.tolist())
        self.config.cfg['Other']['extrinsic_mat'] = str(extrinsic_mat.tolist())
        self.config.save()

    def getExtrinsic(self):
        """Get camera extrinsic parameters

        Returns:
            dict[str, list]: _description_
        """
        return {
            "poses"         : ast.literal_eval(self.config.cfg["Other"].get("poses", "[]")),
            "eye"           : ast.literal_eval(self.config.cfg["Other"].get("eye", "[]")),
            "hmtx"          : ast.literal_eval(self.config.cfg["Other"].get("hmtx", "[]")),
            "projection_mat": ast.literal_eval(self.config.cfg["Other"].get("projection_mat", "[]")),
            "extrinsic_mat" : ast.literal_eval(self.config.cfg["Other"].get("extrinsic_mat", "[]")),
        }

    def clip(self, replay_path:str, start_time:datetime, duration:int):
        """clip video

        Args:
            replay_path (str): loop record directory based on ROOTDIR (Project root)
            start_time (datetime): start timestamp of video (localtime)
            duration (int): length of video

        Returns:
            bytes: raw data of clipped video
        """
        clipper = VideoClipper(f"{ROOTDIR}/{replay_path}")
        save_path = clipper.clip(start_time, duration)
        with open(save_path, 'rb') as file:
            ret = file.read()
        return ret
    
    def getMetricData(self) -> recorder_module.MetricData:
        return self.recorder.getMetricData()

    def isStreaming(self) -> bool:
        return self.recorder.isStreaming

if __name__ == "__main__":
    c = Camera("05320372", "test_host")
    # list available cameras on host
    print(c.getAvailableCameras())