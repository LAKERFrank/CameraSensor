from enum import Enum, auto

class MQTTmsg(dict):
    def __init__(self, source, msg_type, parameter):
        self['source'] = source
        self['msg_type'] = msg_type
        self['parameter'] = parameter

class MsgContract():
    class ID(Enum):
        STOP = auto()
        # System Control
        SYSTEM_CLOSE = auto()
        # Camera Control
        CAMERA_STREAM = auto()
        CALCULATE_CAMERA_EXTRINSIC = auto()
        CAMERA_SETTING = auto()
        # Tracknet
        TRACKNET_DONE = auto()
        # MQTT SUBSCRIBE MESSAGE
        ON_MESSAGE = auto()
        # UI updates state
        PAGE_CHANGE = auto()

    def __init__(self, id, arg:bool=False, value=None, reply=None, request=None, data=None):
        self.id = id
        self.arg = arg
        self.value = value
        self.reply = reply
        self.request = request
        self.data = data

class MqttContract():
    class ID(Enum):
        STOP = auto()
        PUBLISH = auto()
        SUBSCRIBE = auto()

    def __init__(self, id, topic, payload):
        self.id = id
        self.topic = topic
        self.payload = payload