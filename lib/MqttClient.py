import paho.mqtt.client as mqtt
import logging
import json

class MqttClient():
    '''A simple mqtt client generator'''
    def __init__(self):
        
        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_message = self.on_message
        
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
        try: self.mqttc.disconnect()
        except: pass
        self.mqttc.loop_stop()

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        logging.info(f"Connected with result code {reason_code}")
    
    def on_message(self, client, userdata, msg):
        logging.info(f"Received on topic '{msg.topic}': {json.loads(msg.payload)}")


