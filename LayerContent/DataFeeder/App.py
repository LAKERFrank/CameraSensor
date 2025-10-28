import paho.mqtt.client as mqtt
import logging
import ipaddress
import json
import signal

from datetime import datetime


class ExampleAPP():
    
    def __init__(self, broker_ip:str=None, broker_port:str=None):
        self.target_devices_name = "ContentDevice"
        
        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_message = self.on_message
        
        self.CONTROL_PLANE_QOS = 2
        
        if broker_ip == None:
            self.mqttc.connect("host.docker.internal")
            # self.mqttc.connect("localhost")
        else:
            try:
                ipaddress.ip_address(broker_ip)
                self.mqttc.connect(broker_ip, broker_port)
            except:
                logging.error("Fail to connect MQTT Broker.")   
        
        self.mqttc.loop_start()
        
    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        
        print(f"Connected with result code {reason_code}")

        self.mqttc.subscribe(f"/DATA/ContentDevice/ContentLayer/Model3D")
        self.mqttc.subscribe(f"/DATA/ContentDevice/ContentLayer/Model3D/Event")
        self.mqttc.subscribe(f"/DATA/ContentDevice/ContentLayer/Model3D/Segment")
                
    def on_message(self, client, userdata, msg):
        # logging.warning(f"{__class__}.{__name__} unhandled message from MQTT topic: {msg.topic}, payload: {msg.payload[:20]}...")
        
        print(f"[Application] Reveived on topic '{msg.topic}': {json.loads(msg.payload)}")
        
    def stop(self):
        print("Stopping application...")
        try:
                self.mqttc.loop_stop()
                self.mqttc.disconnect()
                print("Application stopped.")
        except Exception as e:
            logging.error(f"Error while stopping applicatipn: {e}")
    
    def run(self):
        pass



        
if __name__ == "__main__":
    app = ExampleAPP()
    
    app.run()
    
    signal.sigwait([signal.SIGTERM, signal.SIGINT])
    app.stop()