import paho.mqtt.client as mqtt
import logging
import ipaddress
import json
import signal
import uuid

from datetime import datetime

from LayerApplication.utils.CameraExplorer import CameraExplorer
from LayerCamera.camera.Rpc_camera import RpcCamera
from LayerSensing.Rpc_sensing import RpcSensing
from LayerContent.RpcContent import RpcContent

class ExampleAPP():
    
    def __init__(self, broker_ip:str=None, broker_port:str=None):
        self.app_uuid = str(uuid.uuid4())
        self.target_devices_name = []
        self.has_explored = False
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_message = self.on_message
        
        self.CONTROL_PLANE_QOS = 2
        
        if broker_ip == None:
            # self.mqttc.connect("host.docker.internal")
            self.mqttc.connect("140.113.213.131", 1883, 60)
        else:
            try:
                ipaddress.ip_address(broker_ip)
                self.mqttc.connect(broker_ip, broker_port)
            except:
                logging.error("Fail to connect MQTT Broker.")   
        
        self.mqttc.loop_start()
        
        self.start()
        
    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        
        print(f"Connected with result code {reason_code}")
               
        if self.has_explored:
            self.subscribeLayers()
                
    def on_message(client, userdata, msg):
        logging.warning(f"{__class__}.{__name__} unhandled message from MQTT topic: {msg.topic}, payload: {msg.payload[:20]}...")
        
        print(f"[ApplicationLayer] Reveived on topic '{msg.topic}': {json.loads(msg.payload)}")
            
    def start(self):
        camera_explore = CameraExplorer(self.mqttc)
        camera_explore.explore(app_uuid = self.app_uuid, date = self.timestamp)
        self.target_devices_name = camera_explore.choose_target_cameras()
        
    def stop(self):
       print("Stopping application...")
       try:
            self.mqttc.loop_stop()
            self.mqttc.disconnect()
            print("Application stopped.")
            exit(0)
       except Exception as e:
           logging.error(f"Error while stopping applicatipn: {e}")
    
    def run(self):
        rpc_camera=RpcCamera(self.target_devices_name[0], self.mqttc)
        rpc_sensing=RpcSensing(self.target_devices_name[0], self.mqttc)
        rpc_content=RpcContent(self.target_devices_name[0], self.mqttc)
        
        cam_0_ks = [[1214.482199462374, 0.0, 739.9438294260159], [0.0, 1208.53844690556, 579.0715103803774], [0.0, 0.0, 1.0]]
        cam_0_poses = [[0.9912996651057628, -0.06131801920511748, 0.11646919971375702], [-0.10714455122731403, -0.4724936268700366, 0.8747970151466546], [0.0013901344160615614, -0.8796650282855345, -0.47495266573075456]]
        cam_0_eye = [[-0.6469120514836701, -2.998389151773412, 2.8509955452830487]]
        cam_0_hmtx = [[0.044402081003816964, -0.010166438048847768, -26.980442037184748], [-0.00546339128059333, -0.058712620800645654, 57.11959390939807], [-2.10913491317734e-05, 0.013415170000639187, 1.0]]
        cam_0_dist = [[-0.5357608528825365, 0.42420392423835374, -0.0020378974867134004, -0.004261218299961821, -0.2375760856910901]]
        cam_0_newcameramtx = [[1213.6387939453125, 0.0, 739.4299903432475], [0.0, 1207.41943359375, 578.535342122268], [0.0, 0.0, 1.0]]
        cam_0_projection_mat = [[1275.6979532328862, 494.4561380309509, -384.4263974512296, 3454.6496152003547], [-21.77107540216067, -115.57454813967456, -1333.6916222420523, 3802.11983993778], [0.09460213035345078, 0.8552864193916321, -0.5094463229179382, 4.263217449188232]]
        cam_0_extrinsic_mat = [[0.993496835231781, -0.11368151009082794, -0.006366398185491562, 0.24908463656902313], [-0.06335971504449844, -0.5055310130119324, -0.8604788780212402, 1.1062418222427368], [0.09460213035345078, 0.8552864193916321, -0.5094463229179382, 4.263217449188232]]
        
        cam_1_ks = [[1716.985921931758, 0.0, 733.9986228296484], [0.0, 1717.9320484971222, 576.6544219941281], [0.0, 0.0, 1.0]]
        cam_1_poses = [[0.9879076106735734, 0.0787023084550984, -0.13358330515850358], [0.07718527752817389, -0.33350717000271657, 0.9395825671486664], [0.029396326955745733, -0.9385314334241711, -0.33554893098010385]]
        cam_1_eye = [[0.9359832488494803, -4.3296505191614765, 2.721944104510576]]
        cam_1_hmtx = [[0.05664382715424806, 0.014891247558090518, -50.848887843219806], [0.008434299348610907, -0.07542958926324395, 63.06438100073595], [-0.0005253957802907401, 0.016519128356882982, 0.9999999999999999]]
        cam_1_dist = [[-0.8858631187090652, 2.557997497124888, -0.01750646305898115, -0.01117379898422549, -4.373555283062699]]
        cam_1_newcameramtx = [[1619.770751953125, 0.0, 772.972176572941], [0.0, 1644.7835693359375, 552.1008150551897], [0.0, 0.0, 1.0]]
        cam_1_projection_mat = [[1543.9331862041768, 858.8709376764804, -315.8391582242493, 4280.333086920458], [92.13512607037094, -296.4593628282531, -1706.9713941770674, 5738.390244796567], [-0.08338155597448349, 0.8734925985336304, -0.4796437919139862, 6.432016849517822]]
        cam_1_extrinsic_mat = [[0.9929706454277039, 0.11340213567018509, 0.03390118479728699, -0.4268733561038971], [0.08400506526231766, -0.47344547510147095, -0.8768081665039062, 1.3298214673995972], [-0.08338155597448349, 0.8734925985336304, -0.4796437919139862, 6.432016849517822]]

        date = 'r_2024-10-23_20-50-38'
        data = [
                {'camera': 'cam_0', 'fps': 120.0, 'parameters': {'ks': cam_0_ks, 'poses': cam_0_poses, 'eye': cam_0_eye, 'dist': cam_0_dist, 'newcameramtx': cam_0_newcameramtx, 'projection_mat': cam_0_projection_mat}},
                {'camera': 'cam_1', 'fps': 120.0, 'parameters': {'ks': cam_1_ks, 'poses': cam_1_poses, 'eye': cam_1_eye, 'dist': cam_1_dist, 'newcameramtx': cam_1_newcameramtx, 'projection_mat': cam_1_projection_mat}}
            ]
        
        # print(rpc_camera.ping())
        print("startModel3D: ",rpc_content.startModel3D(date, data))
        print("startTrackNet: ",rpc_sensing.start_thread("TrackNet_0"))
        
        self.stop()
        
if __name__ == "__main__":
    app = ExampleAPP()
    
    app.run()
    
    signal.sigwait([signal.SIGTERM, signal.SIGINT])
    app.stop()