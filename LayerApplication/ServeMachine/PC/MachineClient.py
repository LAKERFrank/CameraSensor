import os
import queue
import sys
import threading
import time
import json
from enum import Enum, auto
from datetime import datetime
from typing import overload
from multipledispatch  import dispatch
import paho.mqtt.client as mqtt
from PyQt5.QtCore import QThread

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")
from lib.message import MQTTmsg

class MQTTCommunicator(QThread):
    settings = {
        'speed': 0,
        'yaw': 0,
        'pitch': 0,
        'position': {
            'x': 0,
            'y': 0,
            'height': 0
        }
    }
    range = {
        'speed': { 
            'min': 0,
            'max': 0
        },
        'yaw': {
            'min': 0,
            'max': 0
        },
        'pitch': {
            'min': 0,
            'max': 0
        }
    }
    status = None

    class State(Enum):
        Disconnected = auto()
        Preparing = auto()
        Ready = auto()
        Serving = auto()

    def __init__(self, broker_ip, username='', password=''):
        super().__init__()
        self.status = self.State.Disconnected

        # Serve Machine Event
        self.queryEvent = threading.Event()
        self.settingEvent = threading.Event()
        self.servingEvent = threading.Event()

        # Initial
        self.return_message = {}
        self.machine_name = None

        # Event Queue
        self.waitEvent = threading.Event()
        self.messageQueue = queue.Queue(maxsize=10)

        # Settings counter
        self.setting_counter = {
            'speed': 0,
            'yaw': 0,
            'pitch': 0
        }
        self.counterLock = threading.Lock()

        # Setup MQTT
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.username_pw_set(username, password)
        self.client.connect(broker_ip)
        self.stop = False

        self.machine_list = []
        self.start()
        time.sleep(1)

        self.query_topic()
        #self.synchronize_time()

    def on_connect(self, client, userdata, flag, rc):
        client.subscribe('app')

    def on_message(self, client, userdata, msg):
        self.messageQueue.put_nowait(msg)
        self.waitEvent.set()

    def query_topic(self):
        msg = MQTTmsg('app', 'query', 'topic_name')
        self.send_msg('broadcast', msg)
        time.sleep(1)
        return

    def synchronize_time(self):
        now = datetime.now()
        myDatetime = now.strftime("%Y/%m/%d %H:%M:%S")
        msg = MQTTmsg('app', 'synchronize', myDatetime)
        self.send_msg('broadcast', msg)

    def connect(self, machine_name):
        if machine_name in self.machine_list:
            self.machine_name = machine_name
            self.query_status()
            if self.machine_available:
                self.status = self.State.Ready
                return True
            else:
                self.machine_name = None
                return False
        else:
            return False

    def disconnect(self):
        #self.setting(speed=self.range['speed']['min'])
        self.stop = True

    def query_status(self):
        msg = MQTTmsg('app', 'query', 'status')
        self.send_msg(self.machine_name, msg)
        self.queryEvent.wait()
        self.queryEvent.clear()

        return

    def setting(self, speed=None, yaw=None, pitch=None):
        parameter = {}
        self.counterLock.acquire()
        if speed is not None:
            max = self.range['speed']['max']
            min = self.range['speed']['min']
            if speed > max:
                speed = max
                print(f"ERROR:speed out of range {min}~{max}")
            elif speed < min:
                speed = min
                print(f"ERROR:speed out of range {min}~{max}")
            self.settings['speed'] = speed
            self.setting_counter['speed'] += 1
            parameter['speed'] = speed
        if yaw is not None:
            max = self.range['yaw']['max']
            min = self.range['yaw']['min']
            if yaw > max:
                yaw = max
                print(f"ERROR:yaw out of range {min}~{max}")
            elif yaw < min:
                yaw = min
                print(f"ERROR:yaw out of range {min}~{max}")
            self.settings['yaw'] = yaw
            self.setting_counter['yaw'] += 1
            parameter['yaw'] = yaw
        if pitch is not None:
            max = self.range['pitch']['max']
            min = self.range['pitch']['min']
            if pitch > max:
                pitch = max
                print(f"ERROR:pitch out of range {min}~{max}")
            elif pitch < min:
                pitch = min
                print(f"ERROR:pitch out of range {min}~{max}")
            self.settings['pitch'] = pitch
            self.setting_counter['pitch'] += 1
            parameter['pitch'] = pitch
        self.counterLock.release()

        msg = MQTTmsg('app', 'setting', parameter)
        self.send_msg(self.machine_name, msg)

        '''
        self.settingEvent.wait()
        self.settingEvent.clear()
        
        if speed != None:
            print(self.return_message['speed'])
        if yaw != None:
            print(self.return_message['yaw'])
        if pitch != None:
            print(self.return_message['pitch'])
        '''

        return

    def serve(self, balls, interval=2000, delta=None):
        parameter = {
            'balls': balls,
            'interval': interval
        }
        if delta is not None:
            parameter['delta'] = delta

        msg = MQTTmsg('app', 'serve', parameter)
        self.send_msg(self.machine_name, msg)

        self.servingEvent.wait()
        self.servingEvent.clear()

    def writeReg(self, reg, value):
        parameter = {
            'reg': reg,
            'value': value
        }
        msg = MQTTmsg('app', 'writeReg', parameter)
        self.send_msg(self.machine_name, msg)

    def send_msg(self, topic, msg):
        self.client.publish(topic, json.dumps(msg))

    def run(self):
        try:
            self.client.loop_start()
            while True:
                if self.stop:
                    break
                if self.messageQueue.empty():
                    self.waitEvent.wait()
                    self.waitEvent.clear()
                    continue
                else:
                    msg = self.messageQueue.get_nowait()
                    data = json.loads(msg.payload)
                    source = data['source']
                    msg_type = data['msg_type']
                    parameter = data['parameter']
                    if msg_type == 'reply':
                        if 'topic_name' in parameter:
                            if parameter['topic_name'] not in self.machine_list:
                                self.machine_list.append(parameter['topic_name'])
                        elif source == self.machine_name:
                            if 'status' in parameter: #r_status
                                status = parameter['status']
                                self.machine_available = status['available']
                                if self.machine_available:
                                    self.settings = status['settings']
                                    self.range = status['range']
                                    self.queryEvent.clear()
                                self.queryEvent.set()
                            if 'setting' in parameter: #r_done
                                setting = parameter['setting']
                                self.counterLock.acquire()
                                if 'speed' in setting:
                                    self.setting_counter['speed'] -= 1
                                    if setting['speed'] == 'done':
                                        self.return_message['speed'] = f'speed:done'
                                    else:
                                        self.return_message['speed'] = setting['speed']
                                if 'yaw' in setting:
                                    self.setting_counter['yaw'] -= 1
                                    if setting['yaw'] == 'done':
                                        self.return_message['yaw'] = f'yaw:done'
                                    else:
                                        self.return_message['yaw'] = setting['yaw']
                                if 'pitch' in setting:
                                    self.setting_counter['pitch'] -= 1
                                    if setting['pitch'] == 'done':
                                        self.return_message['pitch'] = f'pitch:done'
                                    else:
                                        self.return_message['pitch'] = setting['pitch']
                                if self.setting_counter['speed'] == 0 and self.setting_counter['yaw'] == 0 \
                                    and self.setting_counter['pitch'] == 0:
                                    self.settingEvent.set()
                                self.counterLock.release()
                            if 'serve' in parameter:
                                if parameter['serve'] == 'start':
                                    self.status = self.State.Serving
                                if parameter['serve'] == 'finish':
                                    self.status = self.State.Ready
                                    self.servingEvent.set()
        finally:
            self.client.loop_stop()
            self.client.disconnect()

class MachineClient(MQTTCommunicator):
    def __init__(self, broker_ip, username='', password=''):
        super().__init__(broker_ip, username, password)

    def connect(self, machine_name):
        status = super().connect(machine_name)
        if status:
            print(f'success connect to {self.machine_name}')
        else:
            print(f'fail connect to {machine_name}')
        return status

    def disconnect(self):
        super().disconnect()

    def setSpeed(self, speed:int):
        super().setting(speed=speed)

    def getSpeed(self):
        return self.settings['speed']

    def setHeight(self, height:float):
        pass

    def getHeight(self):
        return self.settings['position']['height']

    def setYaw(self, yaw:float):
        super().setting(yaw=yaw)

    def getYaw(self):
        return self.settings['yaw']

    def setPitch(self, pitch:float):
        super().setting(pitch=pitch)

    def getPitch(self):
        return self.settings['pitch']

    @dispatch(int, int)
    def serve(self, balls:int, interval:int):
        super().serve(balls=balls, interval=interval)

    @dispatch(int, int, int)
    def serve(self, balls:int, interval:int, delta:int):
        if delta == 0:
            delta = None
        super().serve(balls=balls, interval=interval, delta=delta)

    @dispatch(int, int, int, int, int)
    def serve(self, balls:int, interval:int, speed:int, yaw:int, pitch:int):
        super().setting(speed=speed, yaw=yaw, pitch=pitch)
        super().serve(balls=balls, interval=interval)

    def callback(self):
        pass


if __name__ == '__main__':
    client = MachineClient('192.168.50.136')
