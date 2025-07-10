import os
import sys
import logging
import json
import threading
import paho.mqtt.client as mqtt
from datetime import datetime

def sendPerformance(mqtt_client, from_topic, name, action, fids):
    return;
    # timestamp
    timestamp = datetime.now().timestamp()
    # setup kind
    # kind: 0: send, 1: received, 2: section_name start, 3: section_name end
    if action == 'send':
        kind = 0
    elif action == 'receive':
        kind = 1
    elif action == 'start':
        kind = 2
    elif action == 'end':
        kind = 3
    # setup MQTT message
    payload = {'name': name, 'from_topic':from_topic, 'kind':kind, 'fids':fids, 'timestamp':timestamp}
    mqtt_client.publish('performance', json.dumps(payload))

# 0: start, 1: ready, 2: terminated
def sendNodeStateMsg(mqtt_client, node_name, state, page=None):
    #setup MQTT message
    payload = {node_name: state, 'page_name': page}
    mqtt_client.publish('system_status', json.dumps(payload))