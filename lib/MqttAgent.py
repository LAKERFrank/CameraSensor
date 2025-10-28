import paho.mqtt.client as mqtt
# from lib.MqttClient import MqttClient
import logging
import json
import pickle
import signal

def _make_logger(layer):
    color_table = {
        "CameraLayer" : { "recv": "\033[32m", "send": "\033[92m", "general": "\033[95m" },
        "SensingLayer": { "recv": "\033[33m", "send": "\033[93m", "general": "\033[95m" },
        "ContentLayer": { "recv": "\033[34m", "send": "\033[94m", "general": "\033[95m" },
        "ApplicationLayer": { "recv": "\033[36m", "send": "\033[96m", "general": "\033[95m" },
    }
    def _logger(msg, mode="general"):
        c = color_table.get(layer, {}).get(mode, "")
        print(f"{c}[{layer}] {msg}\033[0m")
    return _logger

def _encode_payload(obj):
    try:
        return json.dumps(obj).encode("utf-8"), "json"
    except TypeError:
        return pickle.dumps(obj), "pickle"

def _decode_payload(b: bytes):
    bb = b if isinstance(b, (bytes, bytearray, memoryview)) else (
        b.encode("utf-8") if isinstance(b, str) else bytes(b)
    )
    try:
        return json.loads(bb) # 先試試看是否能用json decode
    except Exception:
        pass
    
    try:
        return pickle.loads(bb) # 再試試 pickle
    except Exception:
        pass
    
    try:
        return bb.decode("utf-8") # 純文字
    except Exception:
        return bb

class MqttAgent:
    def __init__(self, device_name:str, layer:str):
        self.device_name = device_name
        self.layer = layer
        self._logger = _make_logger(layer)
        
        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_disconnect = self.on_disconnect
        self.mqttc.on_message = self.on_message_fallback

        self.control_handler = ControlHandler(device_name, layer)
        self.data_handler = DataHandler(device_name, layer)
        
        self._connected = False
        self.control_handler.attach(self.mqttc, self.is_connected)
        self.data_handler.attach(self.mqttc, self.is_connected)
        
    def start(self, broker_ip=None, broker_port=1883,
              username: str = None, password: str = None, tls: dict = None):
        if username is not None:
            self.mqttc.username_pw_set(username, password)
        if tls is not None:
            self.mqttc.tls_set(**tls)

        self.mqttc.connect(broker_ip or "host.docker.internal", broker_port)
        self.mqttc.loop_start()

    def stop(self):
        try:self.mqttc.disconnect()
        except Exception: pass
        self.mqttc.loop_stop()

    def is_connected(self)->bool:
        return self._connected
    
    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        self._connected = True
        self._logger(f"Connected with result code {reason_code}")
        
        self.control_handler.resubscribe_all()
        self.data_handler.resubscribe_all()
        
    def on_disconnect(self, client:mqtt.Client, userdata, rc):
        self._connected = False
        self._logger(f"Disconnected with result code {rc}")

    def on_message_fallback(self, client:mqtt.Client, userdata, msg):
        try:
            data = _decode_payload(msg.payload)
            self._logger(f"Received on topic '{msg.topic}': {data}", "recv")
        except Exception:
            self._logger(f"Received on topic '{msg.topic}': exception", "recv")


class ControlHandler():
    def __init__(self, device_name:str, layer:str):
        self.device_name = device_name
        self.layer = layer
        self._logger = _make_logger(layer)
        self.CONTROL_QOS = 2
        
        self.client = None
        self._is_connected = lambda: False
        
        self._funcs = []  # [(func_name, func)]
        self._custom_calls = [] # [(topic_filter, func)] for cases like /CALL/AllCamera/...

    def attach(self, client, is_connected_callable):
        self.client = client
        self._is_connected = is_connected_callable

    def register_function(self, func, func_name_override: str = None):
        # Topic的後綴可以選擇使用指定的func_name，或是使用原本function的名稱
        name = func_name_override if isinstance(func_name_override, str) else func.__name__
        self._funcs.append((name, func))
        self._logger(f"[Control] Registered control function {name}")
        
        if self._is_connected():
            self._bind_rpc(name, func)
            
    def register_custom_call(self, call_topic: str, func):
        '''註冊自定義呼叫(可能Topic與預設不一致等因素，故開放開發者自行註冊)'''
        self._custom_calls.append((call_topic, func))
        self._logger(f"[Control] Registered custom call {call_topic}")
        if self._is_connected():
            self._bind_custom(call_topic, func)
            
    def resubscribe_all(self):
        if not any(n == "ping" for n, _ in self._funcs):
            self._funcs.append(("ping", self._ping))
        for name, func in self._funcs:
            self._bind_rpc(name, func)
        for call_topic, func in self._custom_calls:
            self._bind_custom(call_topic, func)

    def _bind_rpc(self, func_name, func):
        call_topic = f"/CALL/{self.device_name}/{self.layer}/{func_name}"
        return_topic  = f"/RETURN/{self.device_name}/{self.layer}/{func_name}"
        
        self.client.message_callback_add(call_topic, self._wrap(return_topic, func))
        self.client.subscribe(call_topic, qos=self.CONTROL_QOS)
        
        self._logger(f"[Control] Subscribed {call_topic}")
        
    def _bind_custom(self, call_topic, func):
        self.client.message_callback_add(call_topic, func)
        self.client.subscribe(call_topic, qos=self.CONTROL_QOS)
        self._logger(f"[Control] Subscribed {call_topic}")

    def _wrap(self, return_topic, func):
        def wrapper(client, userdata, msg):
            self._logger(f"[Control] Received {msg.topic} ({len(msg.payload)}B)", "recv")
            
            try:
                d = _decode_payload(msg.payload)
                res = func(*d.get("args", []), **d.get("kwargs", {}))
            except Exception as e:
                logging.exception("Control func error")
                res = e
                
            bytedata, _ = _encode_payload(res)
            client.publish(return_topic, bytedata, qos=self.CONTROL_QOS)
            
            self._logger(f"Published {return_topic}", "send")
        return wrapper
            
    def _ping(self):
        return "pong"        
            
            
class DataHandler():
    def __init__(self, device_name:str, layer:str, client_id: str = None):
        self.device_name = device_name
        self.layer = layer
        self._logger = _make_logger(layer)
        self.DEFAULT_QOS = 0

        self._pub_topics = {}     # 短名 -> 完整Topic
        self._subs = []          # [(topic_filter, handler, qos)]

        self.client = None
        self._is_connected = lambda: False
        
    def attach(self, client, is_connected_callable):
        self.client = client
        self._is_connected = is_connected_callable
        
    def register_topic(self, short_name:str, suffix_topic:str):
        self._pub_topics[short_name] = f"/DATA/{self.device_name}/{self.layer}/{suffix_topic}"
        self._logger(f"[Data] Registered topic {short_name} -> {self._pub_topics[short_name]}")

    def publish(self, short_name:str, payload, qos=None):

        topic = self._pub_topics.get(short_name)
        if not topic:
            logging.error(f"[Data] Topic {short_name} not registered")
            return
        if isinstance(payload, (bytes, bytearray, memoryview)):
            bytedata = bytes(payload)
        elif isinstance(payload, str):
            bytedata = payload.encode("utf-8")
        else:
            try:
                bytedata = json.dumps(payload).encode("utf-8")
            except Exception:
                bytedata = pickle.dumps(payload)

        self.client.publish(topic, bytedata, qos=self.DEFAULT_QOS if qos is None else qos)
        if short_name != "metrics" and short_name != "heartbeat":
            self._logger(f"Published DATA '{topic}:: {payload}'", "send")
        
    def subscribe(self, topic:str, callback, qos=None):
        
        q = self.DEFAULT_QOS if qos is None else qos
        self._subs.append((topic, callback, q))
        if self._is_connected():
            self.client.message_callback_add(topic, self._wrap(callback))
            self.client.subscribe(topic, qos=q)
            self._logger(f"[Data] Subscribed {topic}")
        else:
            self._logger(f"[Data] Queued subscription {topic}")
            
    def resubscribe_all(self):
        for tf, func, qos in self._subs:
            self.client.message_callback_add(tf, self._wrap(func))
            self.client.subscribe(tf, qos=qos)
            self._logger(f"Subscribed DATA {tf}")

    def _wrap(self, callback_func):
        def wrapper(client, userdata, msg):
            try:
                data = _decode_payload(msg.payload)
                callback_func(data)
            except Exception as e:
                logging.exception(f"Data handler error {e}")
        
        return wrapper

if __name__ == "__main__":
    camera_layer_server = MqttAgent('camera0', 'CameraLayer')
    camera_layer_server.start()
    signal.sigwait([signal.SIGTERM, signal.SIGINT, signal.SIGKILL])

    print("stopping...")
    camera_layer_server.stop()