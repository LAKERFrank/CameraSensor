import logging
import os
import sys
import numpy
from enum import Enum, auto
import time

import gi
gi.require_version("Gst", "1.0")

from gi.repository import GLib, Gst

from datetime import datetime
from lib.common import loadConfig, loadNodeConfig, checkAUXorBUX

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))

def convert_sample_to_numpy(sample):
    buf = sample.get_buffer()
    timestamp = buf.pts / 1000000000
    caps = sample.get_caps()
    mem = buf.get_all_memory()
    success, config = mem.map(Gst.MapFlags.READ)
    if success:
        data = config.data

        bpp = 4
        dtype = numpy.uint8
        if( caps.get_structure(0).get_value('format') == "BGRx" ):
            bpp = 4

        if(caps.get_structure(0).get_value('format') == "GRAY8" ):
            bpp = 1

        if(caps.get_structure(0).get_value('format') == "GRAY16_LE" ):
            bpp = 1
            dtype = numpy.uint16

        img_mat = numpy.ndarray(
            (caps.get_structure(0).get_value('height'),
            caps.get_structure(0).get_value('width'),
            bpp),
            buffer=data,
            dtype=dtype)

        # new version of gst change the return value of mem.map, config is changed into a pointer.
        # therefore, we need to unmap after the process is completed.
        mem.unmap(config)

    return timestamp, img_mat

class CameraReader():
    class STATUS(Enum):
        CLOSE = auto()
        READY = auto()
        STREAMING = auto()
        RECORDING =auto()

    def __init__(self, config, nodename):

        # camera width, height
        self.width = int(config['width'])
        self.height = int(config['height'])

        self.main_config = config
        self.nodename = nodename

        # initialize
        self.initial()
        # open device
        self.openDevice()

    def initial(self):
        main_config_file = f"{ROOTDIR}/config"
        self.main_config = loadNodeConfig(main_config_file, self.nodename)

        # setup
        Gst.init(sys.argv)
        Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)
        self.camera = None
        self.serialnumber = self.main_config['hw_id'] + "-v4l2"
        # self.serialnumber = self.main_config['hw_id']

        #get modelName(AUX or BUX)
        self.modelName = checkAUXorBUX(self.serialnumber)

        # load camera config
        cfg_file = f"{DIRNAME}/config/{self.main_config['hw_id']}.cfg"
        self.cfg = loadConfig(cfg_file)

        # load fps from config
        self.fps = int(self.cfg['Camera']['fps'])

        # camera driver status
        self._status = self.STATUS.CLOSE
        self.timestamp_base = 0

        # frame queue
        self.consumer = None

    def reInitial(self):
        logging.debug(f'reinitialize CameraReader [{self.serialnumber}]...')
        self.close()
        self.initial()
        self.openDevice()

    def openDevice(self):
        if self.fps == None or self.fps < 30:
            self.fps = 60
        try:
            if(self.modelName == "AUX"):
                self.camera = Gst.parse_launch(
                    "tcambin name=source"
                    f" ! video/x-raw,format=BGRx,width=1440,height=1080,framerate={self.fps}/1"
                    " ! queue"
                    " ! videoconvert"
                    " ! appsink name=sink")
            elif(self.modelName == "BUX" and self.fps != 120):
                self.camera = Gst.parse_launch(
                    "tcambin name=source"
                    f" ! video/x-raw,format=BGRx,width=2048,height=1536,framerate={self.fps}/1"
                    " ! queue"
                    " ! videoconvert"
                    " ! appsink name=sink")
            elif(self.modelName == "BUX" and self.fps == 120):
                self.camera = Gst.parse_launch(
                    "tcambin name=source"
                    f" ! video/x-raw,format=BGRx,width=2048,height=1536,framerate=119/1"
                    " ! queue"
                    " ! videoconvert"
                    " ! appsink name=sink")                
        except GLib.Error as error:
            logging.error(f"Error creating camera: {error}")
            raise
        # Quere the source module.
        source = Gst.ElementFactory.make("tcambin")
        serials = source.get_device_serials_backend()
        logging.debug(f"serials: {serials}")
        if self.serialnumber in serials:
            self.source = self.camera.get_by_name("source")
            self.source.set_property("serial", self.serialnumber)
        else:
            return False
        # Query a pointer to the appsink, so we can assign the callback function.
        appsink = self.camera.get_by_name("sink")
        appsink.set_property("max-buffers",10)
        appsink.set_property("drop",0)
        # tell appsink to notify us when it receives an image
        appsink.set_property("emit-signals", True)
        appsink.connect('new-sample', self.onFrameReady)
        self.property_names = self.source.get_tcam_property_names()
        self.camera.set_state(Gst.State.READY)

        logging.info(f"Open device: {self.serialnumber}")
        self.frame_count = 0

        # camera device is ready
        self._status = self.STATUS.READY
        # load configs
        self.loadCameraConfigs()

    def closeDevice(self):
        if self._status != self.STATUS.CLOSE:
            self.camera.set_state(Gst.State.PAUSED)
            self.camera.set_state(Gst.State.NULL)
            self.camera = None
            self._status = self.STATUS.CLOSE

    def loadCameraConfigs(self): # TODO change to load_camera_config ?
        for name, value in self.cfg.items('Camera'):
            self.setProperties(name, value)

    def onFrameReady(self, appsink):
        try:
            #sample = appsink.get_property('last-sample')
            sample = appsink.emit("pull-sample")
            self.frame_count += 1
            self.consumer.try_put_frame(self.frame_count, sample, self.timestamp_base)
        except GLib.Error as error:
            logging.error(f"Error onFrameReady : {error}")
            raise
        return Gst.FlowReturn.OK

    def setConsumer(self, consumer):
        self.consumer = consumer

    def setProperties(self, name, new_value):
        if self._status == self.STATUS.CLOSE:
            return
        # To modify the name to property of camera
        # 'Exposure'
        if name == 'ExposureAuto':
            name = "Exposure Auto"
        elif name == 'ExposureTimeAbs':
            name = "Exposure Time (us)"
            new_value = f"{new_value}000"
        # 'Gain'
        elif name == 'GainAuto':
            name = "Gain Auto"
        # 'White Balance'
        elif name == 'BalanceWhiteAuto':
            name = "Whitebalance Auto"
        elif name == 'BalanceRatioRed':
            name = "Whitebalance Red"
        elif name == 'BalanceRatioBlue':
            name = "Whitebalance Blue"

        if name not in self.property_names:
            logging.warning(f"name: {name}")
            return

        (ret, value,
         min_value, max_value,
         default_value, step_size,
         value_type, flags,
         category, group) = self.source.get_tcam_property(name)

        if not ret:
            logging.warning(f"could not receive value {name}")
            raise

        try:
            if value_type == "enum" :
                if new_value:
                    new_value = "On"
                else:
                    new_value = "Off"
                self.source.set_tcam_property(name, new_value)
            elif value_type == "boolean":
                if new_value == "On":
                    self.source.set_tcam_property(name, True)
                elif new_value == "Off":
                    self.source.set_tcam_property(name, False)
            elif value_type == "integer":
                new_value = int(new_value)
                if new_value > max_value:
                    new_value = max_value
                elif new_value < min_value:
                    new_value = min_value
                elif new_value == value:
                    return
                self.source.set_tcam_property(name, new_value)
        except Exception as e:
            logging.error(e)

    def startStreaming(self, start_timestamp=None):
        if self._status == self.STATUS.READY:
            logging.info(f"Camera {self.serialnumber} start to stream")
            cur_time = datetime.now().timestamp()
            try:
                # wait for specified time to start camera
                if start_timestamp != None:
                    # prevent negative sleep time(crash),
                    # but it means cameras may not start at the same time,
                    # restarting the application is required.
                    start_delta = start_timestamp - cur_time
                    if start_delta < 0:
                        logging.warning(f"start_delta is negative, please restart the application.")
                    time.sleep(max(0, start_delta))

                self.camera.set_state(Gst.State.PLAYING)
                self.timestamp_base = datetime.now().timestamp()
                error = self.camera.get_state(5000000000)
                if error[1] != Gst.State.PLAYING:
                    logging.error(f"Error starting camera {self.serialnumber}")
                    return False
                logging.info(f"{self.serialnumber} Start Streaming... t: {self.timestamp_base}")
                self._status = self.STATUS.STREAMING
            except: # GError as error:
                logging.error(f"Error starting camera: {self.serialnumber}")
                raise
        else:
            logging.warning(f"Camera device {self.serialnumber} don't start")

    def stopStreaming(self):
        if self._status == self.STATUS.STREAMING or self._status == self.STATUS.RECORDING:
            logging.info(f"Camera {self.serialnumber} stop ...")
            self.camera.set_state(Gst.State.PAUSED)
            self.camera.set_state(Gst.State.READY)
            self._status = self.STATUS.READY
            self.consumer = None
        else:
            logging.warning(f"Camera device {self.serialnumber} isn't streaming")

    def close(self):
        self.stopStreaming()
        self.closeDevice()
        self._status = self.STATUS.CLOSE
