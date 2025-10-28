import paho.mqtt.client as mqtt
import time

from lib.Rpc import RemoteProcedureCall

class RpcContent(RemoteProcedureCall):
    def __init__(self, device_name, mqtt_client):
        super().__init__(device_name, "ContentLayer", mqtt_client)

    def startModel3D(self, date, data, mode):
        return self._call_rpc_sync("Model3D/Start", date=date, data=data, mode=mode)
    
    def stopModel3D(self):
        return self._call_rpc_sync("Model3D/Stop")
    
    def dataFeeder(self, file_path):
        return self._call_rpc_sync("Model3D/Feeder", file_path=file_path)

