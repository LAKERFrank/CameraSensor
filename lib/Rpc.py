import paho.mqtt.client as mqtt
import json
import pickle
import threading
import asyncio

class RemoteProcedureCall():
    def __init__(self, device_name:str, layer:str, mqtt_client: mqtt.Client):
        """_summary_

        Args:
            device_name (str): User-defined device name
            layer (str): Must be one of 'CameraLayer', 'SensingLayer'. 'ContentLayer' or 'ApplicationLayer'
            mqtt_client (mqtt.Client): _description_
        """
        self.device_name = device_name
        self.layer = layer
        self.CONTROL_PLANE_QOS = 2
        
        self.mqttc = mqtt_client 

    @staticmethod
    def _pack_payload(args, kwargs):
        try:
            payload = json.dumps({"args":args, "kwargs":kwargs})
        except TypeError:
            payload = pickle.dumps({"args":args, "kwargs":kwargs})
        return payload

    async def _call_rpc_async(self, func_name, timeout=10, *args, **kwargs):
        payload = self._pack_payload(args, kwargs)

        call_topic = f"/CALL/{self.device_name}/{self.layer}/{func_name}"
        return_topic = f"/RETURN/{self.device_name}/{self.layer}/{func_name}"

        res = None

        lock = asyncio.Event()

        loop = asyncio.get_running_loop()

        def callback(client, userdata, msg):
            nonlocal res
            # check if it's json unserializable
            try:
                res = json.loads(msg.payload)
            except ValueError:
                res = pickle.loads(msg.payload)

            loop.call_soon_threadsafe(lock.set)

        self.mqttc.message_callback_add(return_topic, callback)
        self.mqttc.subscribe(return_topic)
        print(f"\033[94m Subscribed {return_topic}\033[0m")

        self.mqttc.publish(call_topic, payload, qos=self.CONTROL_PLANE_QOS)
        print(f"\033[94m Published {call_topic}\033[0m")

        try:
            await asyncio.wait_for(lock.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise Exception("Timeout exceed")
        finally:
            self.mqttc.message_callback_remove(return_topic)
            self.mqttc.unsubscribe(return_topic)
            print(f"\033[94m Unsubscribed {return_topic}\033[0m")

        if issubclass(type(res), Exception):
            raise res

        return res
        
    def _call_rpc_sync(self, func_name, timeout=10, *args, **kwargs):
        payload = self._pack_payload(args, kwargs)

        call_topic = f"/CALL/{self.device_name}/{self.layer}/{func_name}"
        return_topic = f"/RETURN/{self.device_name}/{self.layer}/{func_name}"

        lock = threading.Event()

        res = None

        def callback(client, userdata, msg):
            nonlocal res
            # check if it's json unserializable
            try:
                res = json.loads(msg.payload)
            except ValueError:
                res = pickle.loads(msg.payload)
            lock.set()

        self.mqttc.message_callback_add(return_topic, callback)
        self.mqttc.subscribe(return_topic)
        print(f"\033[94m Subscribed {return_topic}\033[0m")

        self.mqttc.publish(call_topic, payload, qos=self.CONTROL_PLANE_QOS)
        print(f"\033[94m Published {call_topic}\033[0m")

        try:
            if not lock.wait(timeout):
                raise Exception("Timeout exceed")
        finally:
            self.mqttc.message_callback_remove(return_topic)
            self.mqttc.unsubscribe(return_topic)
            print(f"\033[94m Unsubscribed {return_topic}\033[0m")

        if issubclass(type(res), Exception):
            raise res

        return res

    def ping(self, timeout=1):
        self._call_rpc_sync("ping", timeout)
    
    async def ping(self, timeout=1):
        return await self._call_rpc_async("ping", timeout)
