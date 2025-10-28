import json
import time
import logging
import paho.mqtt.client as mqtt
import asyncio

from datetime import datetime

from lib.common import ROOTDIR, loadConfig
from .RpcManager import RpcManager

class RpcStreamingBadminton:
    def __init__(self, manager:RpcManager):
        self.manager = manager
        self.mqttc:mqtt.Client = manager.mqttc

        self.sensing_topics = {}

        for idx, agent in self.manager.sensingLayerAgents.items():
            self.sensing_topics[idx] = f'/DATA/{agent.device_name}/SensingLayer/TrackNet'

        self.content_topic_point = ""
        self.content_topic_event = ""
        self.content_topic_segment = ""

        if self.manager.contentLayerAgent:
            self.content_topic_point = f'/DATA/{self.manager.contentLayerAgent.device_name}/ContentLayer/Model3D/Point'
            self.content_topic_event = f'/DATA/{self.manager.contentLayerAgent.device_name}/ContentLayer/Model3D/Event'
            self.content_topic_segment = f'/DATA/{self.manager.contentLayerAgent.device_name}/ContentLayer/Model3D/Segment'

        self.sensing_callbacks = {}
        self.content_callback_point = None
        self.content_callback_event = None
        self.content_callback_segment = None

    def _verifyRequirements(self):
        if len(self.manager.cameraLayerAgents) < 2:
            raise Exception("We needs at least 2 camera agents")
        if len(self.manager.sensingLayerAgents) < 2:
            raise Exception("We needs at least 2 sensing agents")
        if self.manager.contentLayerAgent is None:
            raise Exception("We needs at least 1 content agents")

    def _func_wrapper(self, func):
        def wrapper(client, userdata, msg):
            return func(msg.payload)

        return wrapper

    def _setupSubscribe(self):
        # setup sensing callback
        for idx, topic in self.sensing_topics.items():
            if (callback := self.sensing_callbacks.get(idx, None)) is not None:
                self.mqttc.subscribe(topic)
                self.mqttc.message_callback_add(topic, self._func_wrapper(callback))
        # setup content callback
        if self.content_topic_point:
            if (callback := self.content_callback_point) is not None:
                self.mqttc.subscribe(self.content_topic_point)
                self.mqttc.message_callback_add(self.content_topic_point, self._func_wrapper(callback))
        if self.content_topic_event:
            if (callback := self.content_callback_event) is not None:
                self.mqttc.subscribe(self.content_topic_event)
                self.mqttc.message_callback_add(self.content_topic_event, self._func_wrapper(callback))
        if self.content_topic_segment:
            if (callback := self.content_callback_segment) is not None:
                self.mqttc.subscribe(self.content_topic_segment)
                self.mqttc.message_callback_add(self.content_topic_segment, self._func_wrapper(callback))

    def _setupUnsubscribe(self):
        for _, topic in self.sensing_topics.items():
            self.mqttc.unsubscribe(topic)
            self.mqttc.message_callback_remove(topic)
        if self.content_topic_point:
            self.mqttc.unsubscribe(self.content_topic_point)
            self.mqttc.message_callback_remove(self.content_topic_point)
        if self.content_topic_event:
            self.mqttc.unsubscribe(self.content_topic_event)
            self.mqttc.message_callback_remove(self.content_topic_event)
        if self.content_topic_segment:
            self.mqttc.unsubscribe(self.content_topic_segment)
            self.mqttc.message_callback_remove(self.content_topic_segment)

    def _readCamConfigForModel3D(self, path):
        cfg = loadConfig(path)

        return {
            "ks": eval(cfg["Other"].get("ks", [])),
            "poses": eval(cfg["Other"].get("poses", [])),
            "eye": eval(cfg["Other"].get("eye", [])),
            "hmtx": eval(cfg["Other"].get("hmtx", [])),
            "dist": eval(cfg["Other"].get("dist", [])),
            "newcameramtx": eval(cfg["Other"].get("newcameramtx", [])),
            "projection_mat": eval(cfg["Other"].get("projection_mat", [])),
            "extrinsic_mat": eval(cfg["Other"].get("extrinsic_mat", [])),
        }

    def testStart(self, read_video = False, mode = ""):

        self._verifyRequirements()

        self._setupSubscribe()

        replay_dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        agent_names = [cameraAgent.device_name for _, cameraAgent in list(self.manager.cameraLayerAgents.items())[:4]]

        feeder_dir = "2025-06-18_13-51-20"

        tracknet_csv = [
            f"/app/replay/{feeder_dir}/TrackNet_0.csv",
            f"/app/replay/{feeder_dir}/TrackNet_1.csv",
            f"/app/replay/{feeder_dir}/TrackNet_2.csv",
            f"/app/replay/{feeder_dir}/TrackNet_3.csv",
        ]

        data = [{"idx":idx, "device":agent_names[idx], "fps": 120.0, "parameters": self._readCamConfigForModel3D(f"/{ROOTDIR}/replay/{feeder_dir}/CameraReader_{idx}.cfg")} for idx in range(4)]

        print(data)

        ret = self.manager.contentLayerAgent.startModel3D(replay_dirname, data, mode)
        logging.info(f"Model3D: {str(ret)}")

        durations = [3]

        if read_video:
            for cam_idx, sensingAgent in list(self.manager.sensingLayerAgents.items())[:4]:
                ret = sensingAgent.startTrackNet((640, 480), "tracknet_v2", "no114_30.tar", replay_dirname, cam_idx)
                logging.info(f"TrackNet_{cam_idx}: {str(ret)}")

            videos = [
                f"/app/replay_sportxai/2025-04-24_16-21-34/CameraReader_0_bright.mp4",
                f"/app/replay_sportxai/2025-04-24_16-21-34/CameraReader_1_bright.mp4",
                f"/app/replay_sportxai/2025-04-24_16-21-34/CameraReader_2_bright.mp4",
                f"/app/replay_sportxai/2025-04-24_16-21-34/CameraReader_3_bright.mp4",
            ]

            for file_idx, (cam_idx, cameraAgent) in enumerate(list(self.manager.cameraLayerAgents.items())[:4]):
                ret = cameraAgent.startVideoFeeder(videos[file_idx])
                durations.append(ret)
        else:
            for i, (cam_idx, sensingAgent) in enumerate(list(self.manager.sensingLayerAgents.items())[:len(tracknet_csv)]):
                print(sensingAgent.device_name)
                ret = sensingAgent.startDatafeeder(filepath=tracknet_csv[i])
                durations.append(ret)

        d = max(durations)

        logging.info(f"{__class__}.{__name__} play for {d:.2f} seconds")

        return d

    def testStop(self, read_video = False):
        if read_video:
            for cam_idx, cameraAgent in list(self.manager.cameraLayerAgents.items())[:4]:
                ret = cameraAgent.stopVideoFeeder()
            for idx, sensingAgent in list(self.manager.sensingLayerAgents.items())[:4]:
                ret = sensingAgent.stopTrackNet()
        else:
            for idx, sensingAgent in list(self.manager.sensingLayerAgents.items())[:4]:
                ret = sensingAgent.stopDatafeeder()
                logging.info(f"TrackNet_{idx}: {str(ret)}")

        ret = self.manager.contentLayerAgent.stopModel3D()
        logging.info(f"Model3D: {str(ret)}")

        self._setupUnsubscribe()

    def start(self, content_mode="CES", tracknet_ver="tracknet_v2", record_mode="h264_high"):
        """Start

        Args:
            content_mode (str, optional): Content 模式. Defaults to "CES".
            tracknet_ver (str, optional): Tracknet 版本. Available options: ["tracknet_v2", "tracknet_1000"].
            record_mode (str, optional): Camera 錄影模式. Available options: ["none", "h264_low", "h264_high"].
        """

        self._verifyRequirements()

        self._setupSubscribe()

        # save dir
        replay_dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # ====== Setup content ======
        data = []
        for cam_idx, cameraAgent in list(self.manager.cameraLayerAgents.items()):
            cam_parameter = cameraAgent.getIntrinsic()
            cam_parameter.update(cameraAgent.getExtrinsic())
            data.append({
                'idx': cam_idx,
                'device': cameraAgent.device_name,
                'fps': 120.0,
                'parameters': cam_parameter
            })
        
        self.manager.contentLayerAgent.startModel3D(date=replay_dirname, data=data, mode=content_mode)

        # ====== Setup sensing ======

        camera_image_res = (512, 288)

        if tracknet_ver == "tracknet_1000":
            camera_image_res = (640, 480)
            # Setup sensing
            for cam_idx, sensingAgent in list(self.manager.sensingLayerAgents.items()):
                cameraAgent = self.manager.cameraLayerAgents[cam_idx]
                ret = sensingAgent.startTrackNet(cameraAgent.resolution, "tracknet_1000", "best.pt", replay_dirname, cam_idx)
                logging.info(f"TrackNet_{cam_idx}: {str(ret)}")
        else:
            # tracknet_v2
            camera_image_res = (512, 288)
            for cam_idx, sensingAgent in list(self.manager.sensingLayerAgents.items()):
                cameraAgent = self.manager.cameraLayerAgents[cam_idx]
                ret = sensingAgent.startTrackNet(cameraAgent.resolution, "tracknet_v2", "no114_30.tar", replay_dirname, cam_idx)
                logging.info(f"TrackNet_{cam_idx}: {str(ret)}")

        # ====== Setup camera ======

        async def startCamera():
            tasks = []
            # Setup camera recording
            for cam_idx, cameraAgent in list(self.manager.cameraLayerAgents.items()):
                tasks.append(
                    cameraAgent.async_startRecording(
                        image_buf=True, image_res=camera_image_res,
                        save_dirname=replay_dirname, cam_idx=cam_idx, mode=record_mode)
                )
            await asyncio.gather(*tasks)

        asyncio.run(startCamera())

    def stop(self):

        # ====== Stop camera ======
        async def stopCamera():
            tasks = []
            # Setup camera recording
            for _, cameraAgent in list(self.manager.cameraLayerAgents.items()):
                tasks.append(
                    cameraAgent.async_stopRecording()
                )
            await asyncio.gather(*tasks)

        asyncio.run(stopCamera())

        # ====== Stop sensing ======
        for idx, sensingAgent in list(self.manager.sensingLayerAgents.items()):
            ret = sensingAgent.stopTrackNet()
            logging.info(f"TrackNet_{idx}: {str(ret)}")

        # ====== Stop content ======
        ret = self.manager.contentLayerAgent.stopModel3D()
        logging.info(f"Model3D_{idx}: {str(ret)}")

        self._setupUnsubscribe()
