'''
    Application Background Service
'''
import json
import logging
import os
import queue
import sys
import threading
import time
import numpy as np

import cv2
import paho.mqtt.client as mqtt
from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal, QRectF, QLineF

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
from lib.common import loadConfig, saveConfig
from lib.message import *
from lib.nodes import CameraReader, Node

# between UI and MQTTService
class SystemService(QThread):
    callback = pyqtSignal(MsgContract)
    def __init__(self, cfg):
        super().__init__()

        # Event Queue
        self.waitEvent = threading.Event()
        self.messageQueue = queue.Queue(maxsize=10)
        # MQTT Service
        broker_ip = cfg['Project']['mqtt_broker']
        broker_port = int(cfg['Project']['mqtt_port'])
        self.mqttService = MQTTService(broker_ip, broker_port)
        self.mqttService.callback.connect(self.on_message)
        # nodes
        self.nodes = {}
        self.tracknet_done = 0

    def stop(self):
        self.sendMessage(MsgContract(id=MsgContract.ID.STOP))
        for node in self.nodes.values():
            node.stop()

    def on_message(self, payload):
        self.sendMessage(MsgContract(MsgContract.ID.ON_MESSAGE, value=payload))

    def sendMessage(self, msg):
        self.messageQueue.put_nowait(msg)
        self.waitEvent.set()

    def handleMessage(self, msg:MsgContract):
        if msg.id == MsgContract.ID.SYSTEM_CLOSE:
            payload = {"stop": True}
            self.mqttService.sendMessage(MqttContract.ID.PUBLISH, 'system_control', payload)
        elif msg.id == MsgContract.ID.ON_MESSAGE:
            data = json.loads(msg.value)
            for node in self.nodes.values():
                if node.name in data:
                    if data[node.name] == 'ready':
                        node.state = Node.State.READY
                    elif data[node.name] == 'terminated':
                        node.state = Node.State.TERMINATED
                    else:
                        logging.error(f"Not supported node state. {node.name}: {data[node.name]}")
            if 'TrackNet_0' in data or 'TrackNet_1' in data or 'TrackNet_2' in data or 'TrackNet_3' in data: # check if all TrackNet Node are finished
                self.tracknet_done = 0
                for node in self.nodes.values():
                    # if node.name.startswith("TrackNet"):
                    #     logging.debug(f'{node.name} status = {node.state}')

                    if (node.name == 'TrackNet_0' and node.state == Node.State.TERMINATED):
                        self.tracknet_done += 1
                    elif (node.name == 'TrackNet_1' and node.state == Node.State.TERMINATED):
                        self.tracknet_done += 1
                    elif (node.name == 'TrackNet_2' and node.state == Node.State.TERMINATED):
                        self.tracknet_done += 1
                    elif (node.name == 'TrackNet_3' and node.state == Node.State.TERMINATED):
                        self.tracknet_done += 1

                if self.tracknet_done == 4:
                    self.delNode("TrackNet_0")
                    self.delNode("TrackNet_1")
                    self.delNode("TrackNet_2")
                    self.delNode("TrackNet_3")
                    msg_emit = MsgContract(id = MsgContract.ID.TRACKNET_DONE, data=data['page_name'])
                    self.callback.emit(msg_emit)
                    self.tracknet_done = 0
                    # break
                elif self.tracknet_done == 2 and data['page_name'] == 'Processing':
                    self.delNode("TrackNet_0")
                    self.delNode("TrackNet_1")
                    # self.delNode("TrackNet_2")
                    # self.delNode("TrackNet_3")
                    msg_emit = MsgContract(id = MsgContract.ID.TRACKNET_DONE, data=data['page_name'])
                    self.callback.emit(msg_emit)
                    self.tracknet_done = False
                    # break
        elif msg.id == MsgContract.ID.PAGE_CHANGE:
                self.callback.emit(msg)
        elif msg.id == MsgContract.ID.CAMERA_STREAM:
            if msg.value == None:
                topic = "cam_control"
                payload = {"streaming": msg.arg}
                for node in self.nodes.values():
                    if node.name == 'Reader':
                        node.isStreaming = False
                self.mqttService.sendMessage(MqttContract.ID.PUBLISH, topic, payload)
            else:
                camera:CameraReader = msg.value
                start_timestamp = msg.data
                if camera.name in self.nodes:
                    topic = camera.name
                    payload = {"streaming": camera.isStreaming, "start_timestamp" : start_timestamp}
                    self.mqttService.sendMessage(MqttContract.ID.PUBLISH, topic, payload)
        elif msg.id == MsgContract.ID.CAMERA_SETTING:
            self.callback.emit(msg)
            camera:CameraReader = msg.value
            payload = msg.data
            if camera.name in self.nodes:
                topic = camera.name
                self.mqttService.sendMessage(MqttContract.ID.PUBLISH, topic, payload)
        else:
            logging.warning(f"{msg.id} is no supported.")

    def addNodes(self, new_nodes):
        if not self.isRunning():
            logging.debug("SystemService is not start up")
            return

        # delete nodes with same name
        for new_node in new_nodes:
            if new_node.name in self.nodes:
                self.delNode(new_node.name)

        for new_node in new_nodes:
            new_node.start()
            self.nodes[new_node.name] = new_node

    def delNode(self, node_name):
        self.nodes[node_name].stop()
        del self.nodes[node_name]

    def run(self):
        try:
            logging.info(f"{self.__class__.__name__}: start up...")
            self.mqttService.start()
            while True:
                if self.messageQueue.empty():
                    self.waitEvent.wait()
                    self.waitEvent.clear()
                try:
                    msg = self.messageQueue.get_nowait()
                    if msg.id == MsgContract.ID.STOP:
                        self.mqttService.stop()
                        break
                    else:
                        self.handleMessage(msg)
                except queue.Empty:
                    logging.warn(f"{self.__class__.__name__}: the message queue is empty.")
        finally:
            if self.mqttService.isRunning() :
                logging.error("error: MQTTService is running")
            logging.info(f"{self.__class__.__name__}: shutdown...")

# Used for communication between UI Services (App.) and system (coachAI)
class MQTTService(QThread):
    callback = pyqtSignal(bytes)
    def __init__(self, broker_ip = 'localhost', broker_port = 1883):
        super().__init__()

        # Event Queue
        self.waitEvent = threading.Event()
        self.messageQueue = queue.Queue(maxsize=10)

        # Setup MQTT
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker_ip, broker_port)

    def on_connect(self, client, userdata, flag, rc):
        logging.info(f"MQTTService: Application Connected with result code: {rc}")
        self.client.subscribe('system_status')

    def on_message(self, client, userdata, msg):
        self.sendMessage(id=MqttContract.ID.SUBSCRIBE, topic=msg.topic, payload=msg.payload)

    def stop(self):
        self.sendMessage(id=MqttContract.ID.STOP)

    def sendMessage(self, id, topic=None, payload=None):
        msg = MqttContract(id, topic, payload)
        self.messageQueue.put_nowait(msg)
        self.waitEvent.set()

    def run(self):
        try:
            self.client.loop_start()
            while True:
                if self.messageQueue.empty():
                    self.waitEvent.wait()
                    self.waitEvent.clear()
                else:
                    msg = self.messageQueue.get_nowait()
                    if msg.id == MqttContract.ID.STOP:
                        break
                    elif msg.id == MqttContract.ID.PUBLISH:
                        logging.debug(f"topic: {msg.topic}, payload: {msg.payload}")
                        self.client.publish(msg.topic, json.dumps(msg.payload))
                    elif msg.id == MqttContract.ID.SUBSCRIBE:
                        self.callback.emit(msg.payload)
        finally:
            self.client.loop_stop()
            logging.info("MQTTService: Application disconnected...")