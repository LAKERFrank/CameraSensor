#!/usr/bin/env python3

# Copyright 2017 The Imaging Source Europe GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# This example will show you how to list information about the available devices
#

import sys
import gi
import subprocess
import logging
import threading
import signal

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")

from gi.repository import GLib, Gst

from lib.common import ROOTDIR
from lib.GracefulKiller import GracefulKiller

logging.getLogger().setLevel(logging.INFO)

class CameraList:

    TERM_TIMEOUT=10

    def __init__(self):
        self.processList:dict[str, subprocess.Popen] = {}
    def add(self, serial):
        if serial in self.processList:
            return
        
        name = f"tcam-{serial}"
        cmd = [
            "python3",
            f"{ROOTDIR}/main_device.py",
            name,
            serial
        ]
        p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        print(f"[Device Monitor] Started supprocess \"{name}\"")
        self.processList[serial] = p
    def remove(self, serial:str):
        p = self.processList.pop(serial)
        p.send_signal(signal.SIGTERM)

        try:
            retcode = p.wait(timeout=self.TERM_TIMEOUT)
            logging.info(f"[Device Monitor] Subprocess serial={serial} safely exited. (return code = {retcode})")
        except subprocess.TimeoutExpired:
            p.kill()
            logging.error(f"[Device Monitor] Subprocess serial={serial} timeout expired, being killed.")

    def clear(self):
        for _, p in self.processList.items():
            p.send_signal(signal.SIGTERM)

        for _, p in self.processList.items():
            p.wait()

class DeviceMonitor:
    def __init__(self):
        Gst.init(sys.argv)  # init gstreamer
        Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)

        self.monitor = Gst.DeviceMonitor.new()

        self.monitor.add_filter("Video/Source/tcam")

        bus = self.monitor.get_bus()
        bus.add_watch(GLib.PRIORITY_DEFAULT, self.bus_function, None)

        self.monitor.start()
        print("Now listening to device changes. Disconnect your camera to see a remove event. Connect it to see a connect event. Press Ctrl-C to end.\n")

        self.camList = CameraList()

        for device in self.monitor.get_devices():
            self.callback_device_added(device)

    def loadUdevRule(self, model, serial):
        model = model.replace(" ", "_")
        command = [
            "sudo",
            "tcam-uvc-extension-loader",
            "-f", "/usr/share/theimagingsource/tiscamera/uvc-extension/usb37.json",
            "-d", f"/dev/v4l/by-id/usb-The_Imaging_Source_Europe_GmbH_{model}_{serial}-video-index0"
        ]
        subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)

    def getDevices(self):
        ret = []
        for device in self.monitor.get_devices():
            ret.append(self.getDeviceInfo(device))
        return ret

    def getDeviceInfo(self, device):
        # struc is a Gst.Structure
        struc = device.get_properties()

        return {
            "model": struc.get_string("model"),
            "serial": struc.get_string("serial"),
            "type": struc.get_string("type")
        }
                                                        
    def callback_device_added(self, device):
        device_info = self.getDeviceInfo(device)
        if device_info["type"] != 'v4l2':
            return
        logging.info(f"[Device Connected] {device_info['serial']} {device_info['model']}")

        # load udev rules
        #self.loadUdevRule(device_info["model"], device_info["serial"])

        self.camList.add(device_info['serial'])

    def callback_device_removed(self, device):
        device_info = self.getDeviceInfo(device)
        if device_info["type"] != 'v4l2':
            return
        logging.info(f"[Device Removed] {device_info['serial']} {device_info['model']}")

        self.camList.remove(device_info['serial'])

    def bus_function(self, bus, message, user_data):
        """
        Callback for the GstBus watch
        """

        if message.type == Gst.MessageType.DEVICE_ADDED:
            device = message.parse_device_added()
            self.callback_device_added(device)
        elif message.type == Gst.MessageType.DEVICE_REMOVED:
            device = message.parse_device_removed()
            self.callback_device_removed(device)

        return True

    def mainLoop(self):
        # This is simply used to wait for events or the user to end this script
        loop = GLib.MainLoop.new(None, False)
        try:
            # needs glib mainloop for callback
            def glib_mainloop():
                loop.run()
            t = threading.Thread(target=glib_mainloop)
            t.start()
            GracefulKiller().wait()
        finally:
            # has to be called when gst_device_monitor_start has been called
            self.monitor.stop()
            self.camList.clear()
            loop.quit()

if __name__ == "__main__":
    monitor = DeviceMonitor()
    monitor.mainLoop()
