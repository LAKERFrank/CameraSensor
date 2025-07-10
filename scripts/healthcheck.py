import argparse
from LayerApplication.utils.Mqtt import MqttClient
from lib.Rpc import RemoteProcedureCall
from lib.common import loadConfig, ROOTDIR

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("hostname")
    parser.add_argument("layer")

    args = parser.parse_args()

    # loading project config
    cfg_file = f"{ROOTDIR}/config"
    cfg = loadConfig(cfg_file)
    broker_ip = cfg["Project"]["mqtt_broker"]
    broker_port = int(cfg["Project"]["mqtt_port"])

    mqtt = MqttClient(broker_ip, broker_port)

    rpc = RemoteProcedureCall(args.hostname, args.layer, mqtt.mqttc)
    try:
        rpc.ping()
        exit(0)
    except Exception as e:
        print(str(e))
        exit(1)
