#!/usr/bin/env python3
import os
import sys
import docker.errors
import gi
import docker
import logging
import threading
from docker.types import DeviceRequest
from lib.GracefulKiller import GracefulKiller

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
gi.require_version("Tcam", "1.0")

from gi.repository import GLib, Gst

logging.getLogger().setLevel(logging.INFO)

class Container:
    def __init__(self, container, name, cam_type, serial):
        self.container = container
        self.name = name
        self.cam_type = cam_type # tcam
        self.serial = serial

class ContainerManager:
    def __init__(self, cams = []):
        logging.info(f"{__class__} init()")
        self.HOST_ROOTDIR = os.getenv("PROJECT_HOST_DIR")
        self.containers:dict[str:Container] = {} # daemon_name -> Container object
        self.client = docker.from_env()

        connected_usb_serials = set([d['serial'] for d in cams])

        # Get connected containers
        for c in self.client.containers.list(all=True, filters={"label":"tw.edu.nycu.cs.nol.camerasensor.cam.type=tcam-usb"}):
            logging.info(f"{__class__} found container \"{c.name}\" exists.")

            serial = c.labels["tw.edu.nycu.cs.nol.camerasensor.cam.serial"]

            if serial in connected_usb_serials:
                connected_usb_serials.remove(serial)
                self.containers[c.name] = Container(c, c.name, "tcam-usb", serial)
                if c.status != "running":
                    c.start()
            else:
                # camera is not connected => remove container
                self.remove(serial)
        # Create
        for serial in connected_usb_serials:
            self.create(serial)

    def getDaemonName(self, serial, cam_type):
        return f"device-{cam_type}-{serial}"

    def _createContainer(self, name, cam_type, serial):
        c = self.client.containers.run(
            image="gitlab.nol.cs.nycu.edu.tw:5050/marvin/camerasensor/camera-sensor:v1.3",
            environment=[
	            "NVIDIA_DRIVER_CAPABILITIES=all",
                "PYTHONPATH=/app"],
            volumes=[
                f"{self.HOST_ROOTDIR}:/app/"],
            command=f"python3 /app/main_device.py {name} {serial}",
            name=name,
            extra_hosts=["host.docker.internal:host-gateway"],
            privileged=True,
            shm_size="8g",
            restart_policy={"Name": "on-failure", "MaximumRetryCount": 5},
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            labels={
                "tw.edu.nycu.cs.nol.camerasensor.cam.type": cam_type,
                "tw.edu.nycu.cs.nol.camerasensor.cam.serial": serial,
            },
            detach=True)

        return Container(c, name, cam_type, serial)

    def create(self, serial, cam_type="tcam-usb"):

        daemon_name = self.getDaemonName(serial, cam_type)

        if daemon_name in self.containers.keys():
            logging.info(f"Container {daemon_name} exists.")
            return

        try:
            c = self._createContainer(daemon_name, cam_type, serial)
            logging.info(f"Container {daemon_name} created.")

            self.containers[daemon_name] = c
        except Exception as e:
            logging.error(e)
            pass

    def remove(self, serial, cam_type="tcam-usb"):
        daemon_name = self.getDaemonName(serial, cam_type)

        try:
            c = self.containers.pop(daemon_name)
            logging.info(f"Removing Container {daemon_name} ...")
            c.container.stop()
            c.container.remove()
            logging.info(f"Removed Container {daemon_name} ...")
        except Exception as e:
            logging.error(e)

    def clear(self):
        logging.info(f"{__class__} clear()")
        c_list = list(self.containers.values())
        for c in c_list:
            self.remove(c.serial, c.cam_type)

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

        self.container_manager = ContainerManager(self.getDevices())

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

        self.container_manager.create(device_info['serial'])

    def callback_device_removed(self, device):
        device_info = self.getDeviceInfo(device)
        if device_info["type"] != 'v4l2':
            return
        logging.info(f"[Device Removed] {device_info['serial']} {device_info['model']}")

        self.container_manager.remove(device_info['serial'])

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
        try:
            # needs glib mainloop for callback
            def glib_mainloop():
                loop = GLib.MainLoop.new(None, False)
                loop.run()
            t = threading.Thread(target=glib_mainloop)
            t.start()
            GracefulKiller().wait()
        finally:
            # has to be called when gst_device_monitor_start has been called
            self.monitor.stop()
            self.container_manager.clear()


if __name__ == "__main__":
    monitor = DeviceMonitor()
    monitor.mainLoop()