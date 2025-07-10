import paho.mqtt.client as mqtt
import ipaddress
import logging
import json
import pickle
import signal

class MqttAgent():
    
    def __init__(self, device_name:str, layer:str):
        """
        Args:
            device_name (str): User-defined device name
            layer (str): Must be one of 'CameraLayer', 'SensingLayer'. 'ContentLayer' or 'ApplicationLayer'
        """
        self.device_name = device_name
        self.layer = layer
        
        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_message = self.on_message
        
        self.CONTROL_PLANE_QOS = 2
        
    def start(self, broker_ip:str=None, broker_port:int=1883):
        if broker_ip == None:
            self.mqttc.connect("host.docker.internal")
        else:
            try:
                self.mqttc.connect(broker_ip, broker_port)
            except:
                logging.error("Fail to connect MQTT Broker.")   
        
        self.mqttc.loop_start()

    def stop(self):
        self.mqttc.loop_stop()
        
    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        self._logger(f"Connected with result code {reason_code}")

        self._add_func_callback(self._ping, "ping")
    
    def on_message(self, client, userdata, msg):
        self._logger(f"Reveived on topic '{msg.topic}': {json.loads(msg.payload)}")
        
    def _add_func_callback(self, func, func_name_override:str=None):
        func_name = func_name_override if isinstance(func_name_override, str) else func.__name__

        call_topic = f"/CALL/{self.device_name}/{self.layer}/{func_name}"
        return_topic = f"/RETURN/{self.device_name}/{self.layer}/{func_name}"

        self.mqttc.message_callback_add(call_topic, self._func_wrapper(return_topic, func))

        self.mqttc.subscribe(call_topic)
        self._logger(f"Subscribed {call_topic}")

    def _logger(self, msg, mode="general"):
        color_table = {
            "CameraLayer" : { "recv": "\033[32m", "send": "\033[92m", "general": "\033[95m" },
            "SensingLayer": { "recv": "\033[33m", "send": "\033[93m", "general": "\033[95m" },
            "ContentLayer": { "recv": "\033[34m", "send": "\033[94m", "general": "\033[95m" },
        }

        try:
            c = color_table[self.layer][mode]
        except:
            c = ""
        
        print(f"{c}[{self.layer}] {msg}\033[0m")

    def _func_wrapper(self, return_topic, func):
        def wrapper(client, userdata, msg):

            self._logger(f"Received from topic {msg.topic} with data {msg.payload}", "recv")

            try:
                data = json.loads(msg.payload)
            except UnicodeDecodeError:
                data = pickle.loads(msg.payload)

            try:
                res = func(*data["args"], **data["kwargs"])
            except Exception as e:
                logging.error(str(e))
                res = e

            # checking if it's json serializable
            try:
                bytedata = json.dumps(res)
            except TypeError as e:
                bytedata = pickle.dumps(res)

            client.publish(return_topic, bytedata, self.CONTROL_PLANE_QOS)

            self._logger(f"Published to topic '{return_topic}", "send")

        return wrapper

    def _ping(self):
        return "pong"
    
if __name__ == "__main__":
    camera_layer_server = MqttAgent('camera0', 'CameraLayer')
    camera_layer_server.start()
    signal.sigwait([signal.SIGTERM, signal.SIGINT, signal.SIGKILL])

    print("stopping...")
    camera_layer_server.stop()