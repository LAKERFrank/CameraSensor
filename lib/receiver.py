import sys
import os
import logging
import json
import queue
import threading
import time
from datetime import datetime
import paho.mqtt.client as mqtt

from .frame import Frame
from .point import Point
from .common import insertById

class RawImgReceiver(threading.Thread):
    def __init__(self, broker, topic, queue_size, callback):
        super().__init__()
        self.killswitch = threading.Event()
        self.callback = callback

        self.topic = topic
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(broker)
        self.client = client

        self.queue = []
        self.queue_size = queue_size

    def try_put_frame(self, data):
        if len(self.queue) < self.queue_size :
            frame = Frame(data['id'], data['timestamp'], data['raw_data'])
            insertById(self.queue, frame)
            # if not self.callback.is_set():
            self.callback.set()
        else:
            # discard 1 second data
            del self.queue[0:30]
            logging.warning("Receiver [{}] is full.".format(self.topic))

    def run(self):
        logging.debug("Receiver [{}] started.".format(self.topic))
        # start
        try:
            self.client.loop_start()
            logging.info("Receiver [{}] is reading".format(self.topic))
            self.killswitch.wait()
        finally:
            self.client.loop_stop()
        # end
        logging.info("Receiver [{}] terminated".format(self.topic))

    def on_connect(self, client, userdata, flag, rc):
        logging.info(f"Receiver {self.topic} Connected with result code: {rc}")
        self.client.subscribe(self.topic)

    def on_message(self, client, userdata, msg):
        data = json.loads(msg.payload)
        self.try_put_frame(data)

    def stop(self):
        self.killswitch.set()

    def set_topic(self, topic, clear=True):
        if self.topic is not None:
            self.client.unsubscribe(self.topic)
        self.client.subscribe(topic)
        self.topic = topic
        if clear:
            self.queue.clear()

class PointReceiver(threading.Thread):
    def __init__(self, client, triangulation_thread, name, in_name, broker, topic, queue_size):
        super().__init__()
        self.killswitch = threading.Event()

        self.triangulation_thread = triangulation_thread
        self.name = name
        self.in_name = in_name 
        self.topic = topic

        # client = mqtt.Client()
        # client.on_connect = self.on_connect
        # client.on_message = self.on_message
        # client.connect(*broker)
        # self.client = client

        self.client = client
        self.client.subscribe(self.topic)
        self.client.message_callback_add(self.topic, self.on_message)

        self.queue = []
        self.queue_size = queue_size
        self.point_count = -1
        logging.debug(f"topic: {self.topic}, queue size:{queue_size} ")

    def try_put_point(self, data):
        if "EOF" in data and data["EOF"]:
            print(f'Receiver {self.in_name} EOF')
            self.triangulation_thread.onReceiverFinished()
        elif "EOS" in data and data["EOS"]:
            print(f'Receiver {self.in_name} EOS')
            self.triangulation_thread.onReceiverFinished()
        else:
            if len(self.queue) < self.queue_size :
                self.point_count = len(data['linear'])
                for i in range(len(data['linear'])):
                    point = Point(fid=data['linear'][i]['id'],
                                timestamp=data['linear'][i]['timestamp'],
                                visibility=data['linear'][i]['visibility'],
                                x=data['linear'][i]['pos']['x'],
                                y=data['linear'][i]['pos']['y'],
                                z=data['linear'][i]['pos']['z'],
                                event=data['linear'][i]['event'],
                                speed=data['linear'][i]['speed'])
                    insertById(self.queue, point)
                    # logging.debug("Receiver [{}] id:{}, ({:>2.3f}, {:>2.3f}, {:>2.3f}), timestamp:{}".format(self.topic, point.fid, point.x, point.y, point.z, point.timestamp))
                    # print(f'[insert] {self.topic} :', data['linear'][i]['id'])
                    # for q in self.queue:
                    #     print(self.queue[q].fid, end='')
                    # print()
            else:
                logging.warning("Receiver [{}] is full.".format(self.topic))

    def run(self):
        logging.debug("Receiver [{}] started.".format(self.topic))
        # start
        try:
            self.client.loop_start()
            logging.info("Receiver [{}] is reading".format(self.topic))
            self.killswitch.wait()
        finally:
            print('Stop receiver: {}'.format(self.topic))
            # self.client.loop_stop()

        # end
        logging.info("Receiver [{}] terminated".format(self.topic))

    def on_connect(self, client, userdata, flag, rc):
        logging.info(f"Point Receiver {self.topic} Connected with result code: {rc}")
        self.client.subscribe(self.topic)

    def on_message(self, client, userdata, msg): #on_tracknet_message
        # print()
        # print('[Point Receiver] Receive TrackNet on topic', msg.topic)
        # print(msg.payload)
        data = json.loads(msg.payload)
        self.try_put_point(data)
        #fids = []
        #for i in range(len(data['linear'])):
        #    fids.append(data['linear'][i]['id'])
        #sendPerformance(self.client, self.topic, self.name, 'receive', fids)
        #sendPerformance(self.client, self.name, 'total', 'start', fids)

    def stop(self):
        self.killswitch.set()

    def clear(self):
        self.queue.clear()