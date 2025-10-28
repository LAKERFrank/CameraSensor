import configparser
import logging
import asyncio
import time

from lib.Rpc import RemoteProcedureCall
from LayerCamera.camera.RpcCamera import RpcCamera
from LayerSensing.RpcSensing import RpcSensing
from LayerContent.RpcContent import RpcContent

class RpcManager:
    def __init__(self, mqttc):

        self.mqttc = mqttc

        self.contentLayerAgent:RpcContent = None
        self.sensingLayerAgents:'dict[int, RpcSensing]' = {}
        self.cameraLayerAgents:'dict[int, RpcCamera]' = {}

        self.contentDeviceName = ""
        self.sensingDeviceNames:'dict[int, str]' = {}
        self.cameraDeviceNames:'dict[int, str]' = {}

        self.cameraConfigs = {}

    def setDeviceNamesFromConfig(self, cfg:configparser.ConfigParser):
        self.contentDeviceName = "ContentDevice"
        cam_names = {}
        for node_name, node_info in cfg.items():
            if node_name[:12] == "CameraReader" and node_info["camerasensor_hostname"] != 'None':
                index = int(node_name[13:])
                cam_names[index] = node_info["camerasensor_hostname"]
                self.cameraConfigs[index] = node_info
                
        self.cameraDeviceNames = cam_names
        self.sensingDeviceNames = cam_names

    def open(self):
        asyncio.run(self._createAllAgents())
        asyncio.run(self._startCamera())

        # sorted by camera index
        self.cameraLayerAgents = dict(sorted(self.cameraLayerAgents.items()))
        self.sensingLayerAgents = dict(sorted(self.sensingLayerAgents.items()))

    def close(self):
        asyncio.run(self._stopCamera())

    async def _createAllAgents(self):
        tasks = []
        for idx, name in self.cameraDeviceNames.items():
            tasks.append(self._createAgent(RpcCamera(name, self.mqttc), idx))
        for idx, name in self.sensingDeviceNames.items():
            tasks.append(self._createAgent(RpcSensing(name, self.mqttc), idx))
        if name := self.contentDeviceName:
            tasks.append(self._createAgent(RpcContent(name, self.mqttc)))

        await asyncio.gather(*tasks)

    async def _createAgent(self, agent:RemoteProcedureCall, idx:int=0):
        if await self._liveCheck(agent):
            if isinstance(agent, RpcCamera):
                self.cameraLayerAgents[idx] = agent
            elif isinstance(agent, RpcSensing):
                self.sensingLayerAgents[idx] = agent
            else:
                self.contentLayerAgent = agent

    async def _startCamera(self):
        tasks = [agent.async_init(software_trigger=False, resync=True) for _, agent in self.cameraLayerAgents.items()]
        await asyncio.gather(*tasks)
        logging.info("All camera initialized (ready).")

        t_ns = time.time_ns()

        tasks = [agent.async_start(t_ns) for _, agent in self.cameraLayerAgents.items()]
        await asyncio.gather(*tasks)

    async def _stopCamera(self):
        tasks = [agent.async_release() for _, agent in self.cameraLayerAgents.items()]
        await asyncio.gather(*tasks)

    async def _liveCheck(self, agent:RemoteProcedureCall):
        try:
            await agent.ping()
            print(f"{type(agent).__name__} \"{agent.device_name}\" is alive.")
            return True
        except Exception:
            print(f"{type(agent).__name__} \"{agent.device_name}\" is not alive.")
            return False
