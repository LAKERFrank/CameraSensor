import paho.mqtt.client as mqtt
import logging
import ipaddress
import json
import signal
import time
import csv

from datetime import datetime


class DataFeeder():
    
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
                
    def on_message(self, client, userdata, msg):
        # logging.warning(f"{__class__}.{__name__} unhandled message from MQTT topic: {msg.topic}, payload: {msg.payload[:20]}...")
        
        print(f"[ContentLayer] Reveived on topic '{msg.topic}': {json.loads(msg.payload)}")
        
    def stop(self):
        print("Stopping application...")
        try:
                self.mqttc.loop_stop()
                self.mqttc.disconnect()
                print("Application stopped.")
        except Exception as e:
            logging.error(f"Error while stopping applicatipn: {e}")

    
    def _time_diff(self, t1: str, t2: str) -> float:
        fmt = "%H:%M:%S.%f"
        try:
            t1_dt = datetime.strptime(t1, fmt)
            t2_dt = datetime.strptime(t2, fmt)
            return max((t2_dt - t1_dt).total_seconds(), 0)
        except Exception as e:
            print(f"[WARN] Time parse failed: {e}")
            return 0.2
    
    
    def run(self):
        time.sleep(0.2)

        file_path = "Content_Output_Sample.csv"

        event_topic = "/DATA/ContentDevice/ContentLayer/Model3D/Event"
        segment_topic = "/DATA/ContentDevice/ContentLayer/Model3D/Segment"

        with open(file_path, 'r', newline='', encoding='utf-8-sig') as csvFile:
            rows = list(csv.DictReader(csvFile))
            has_time_col = 'publish_time' in rows[0]

            for i, row in enumerate(rows):                
                if row.get('Type') == 'Event':
                    event_payload = {
                        'fid': int(row['fid']),
                        'event': int(row['event']),
                        'timestamp': float(row['timestamp']),
                        'position': json.loads(row['position'])
                    }
                    self.mqttc.publish(event_topic, json.dumps(event_payload))
                    print(f"[ContentLayer] Published on topic '{event_topic}': {event_payload}")

                elif row.get('Type') == 'Segment':
                    segment_payload = {
                        'start_fid': int(row['start_fid']),
                        'start_timestamp': float(row['start_timestamp']),
                        'start_position': json.loads(row['start_position']),
                        'end_fid': int(row['end_fid']),
                        'end_timestamp': float(row['end_timestamp']),
                        'end_position': json.loads(row['end_position']),
                        'speed': json.loads(row['speed']),
                        'type': row['ball_type']
                    }
                    self.mqttc.publish(segment_topic, json.dumps(segment_payload))
                    print(f"[ContentLayer] Published on topic '{segment_topic}': {segment_payload}")

                # Handle delay
                if has_time_col and i + 1 < len(rows):
                    current_time = row['publish_time']
                    next_time = rows[i + 1]['publish_time']
                    delay = self._time_diff(current_time, next_time)
                    time.sleep(delay)
                else:
                    time.sleep(0.2)
        
if __name__ == "__main__":
    app = DataFeeder()
    
    app.run()
    
    signal.sigwait([signal.SIGTERM, signal.SIGINT])
    app.stop()