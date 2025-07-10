import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime
from lib.Rpc import RemoteProcedureCall

class RpcCamera(RemoteProcedureCall):
    """RpcCamera

    Example (Using software trigger)::

        c = RpcCamera("debug-cam", client)
        c.init(software_trigger=True)
        c.start()

    Example (Using resync mechanism)::

        c = RpcCamera("debug-cam", client)
        c.init(software_trigger=False)
        t = time.time_ns()
        c.start(t)
        c.setMaster("master-cam")

    """

    def __init__(self, device_name:str, mqtt_client: mqtt.Client):
        """
        Args:
            device_name (str): Device name
            mqtt_client (mqtt.Client): MQTT client
        """
        super().__init__(device_name, "CameraLayer", mqtt_client)

    async def async_init(self, software_trigger:bool=False, resync:bool=False) -> bool:
        ret = await self._call_rpc_async(self.init.__name__, software_trigger=software_trigger, resync=resync)
        self.__updateParameter()
        return ret

    def init(self, software_trigger:bool=False, resync:bool=False) -> bool:
        """相機初始化

        Args:
            software_trigger (bool, optional): 是否要開啟軟體快門. Defaults to False.

        Returns:
            bool: True
        """
        ret = self._call_rpc_sync(self.init.__name__, software_trigger=software_trigger, resync=resync)
        self.__updateParameter()
        return ret

    async def async_start(self, timestamp_ns:int) -> None:
        """開始相機串流

        Args:
            timestamp_ns (int, optional): unix epoch in nanoseconds. 給一個同步的時間，軟體快門會依據此時間同步.

        Returns:
            None:
        """
        ret = await self._call_rpc_async("start", timestamp_ns=timestamp_ns)
        return ret

    async def async_resync(self, timestamp_ns:int, dt_ns:int=0) -> None:
        """重新同步快門時間

        Args:
            timestamp_ns (int): unix epoch in nanoseconds. 同步機制會依照此時間做同步.

        Returns:
            None:
        """
        ret = await self._call_rpc_async("resync", timestamp_ns=timestamp_ns, dt_ns=dt_ns)
        return ret

    def resync(self, timestamp_ns:int, dt_ns:int):
        return self._call_rpc_sync("resync", timestamp_ns=timestamp_ns, dt_ns=dt_ns)

    def __updateParameter(self):
        """Update class properties
        """
        p = self.getCameraParameters()
        self.direction:int = p["direction"]
        self.resolution:'tuple[int,int]' = tuple(p["RecordResolution"])
        self.fps:int = p["fps"]

    async def async_release(self):
        await self._call_rpc_async(self.release.__name__)
        return True

    def release(self):
        """關閉相機

        Returns:
            bool: True
        """
        return self._call_rpc_sync(self.release.__name__)

    def getCaptureFormats(self):
        """取得相機支援的攝影解析度

        Returns:
            list[dict]: 支援的攝影解析度

            Example::

                [
                    {
                        'width': 1440,
                        'height': 1080,
                        'skipping': '1x1',
                        'target_fps': [236, 120, 60, 30]
                    },
                    {
                        'width': 640,
                        'height': 480,
                        'skipping': '2x2',
                        'target_fps': [600, 480, 240, 120, 60, 30]
                    }
                ]
        """
        return self._call_rpc_sync(self.getCaptureFormats.__name__)

    def setCaptureFormat(self, width:int=None, height:int=None, fps:int=None, skipping:str=None, direction:int=None) -> None:
        """設定相機解析度

        Args:
            width (int, optional): Image Width. Defaults to None.
            height (int, optional): Image Height. Defaults to None.
            fps (int, optional): Frames per second. Defaults to None.
            skipping (str, optional): Skipping. Defaults to None.
            direction (int, optional): Direction. Defaults to None.

        Returns:
            None:
        """
        ret = self._call_rpc_sync(self.setCaptureFormat.__name__, width=width, height=height, fps=fps, skipping=skipping, direction=direction)
        self.__updateParameter()
        return ret

    def getDeviceInfo(self):
        """取得相機的相關資訊

        Returns:
            dict:相機資訊

            Example::

                {
                    'brand': 'Image_Source',
                    'model': 'DFK 37AUX273',
                    'serial': '08324005'
                }
        """
        return self._call_rpc_sync(self.getDeviceInfo.__name__)

    def setCameraParameters(self, values:'dict[str, int|float|str]') -> None:
        """設定相機參數

        Args:
            values (dict[str, int|float|str]): 相機參數名稱與數值

                example: {"BlackLevel": 0,"ExposureAuto": "Off","Gain" = 20.0}

        Returns:
            None:
        """
        return self._call_rpc_sync(self.setCameraParameters.__name__, values=values)

    def getCameraParameters(self):
        """取得相機參數

        Returns:
            dict[str, int|float|str|tuple]: 相機參數

            Example::

                {
                    'BlackLevel': 0.0,
                    'ExposureTime': 3012.0,
                    'Gain': 20.0,
                    'BalanceWhiteRed': 1.40625,
                    'BalanceWhiteBlue': 3.0,
                    'BalanceWhiteGreen': 1.015625,
                    'direction': 0,
                    'RecordResolution': (640, 480),
                    'fps': '120',
                    'skipping': '2x2'
                }

        """
        return self._call_rpc_sync(self.getCameraParameters.__name__)

    async def async_startRecording(self, save_dirname:str, cam_idx:int,
                                   image_buf:bool=True, image_res:'tuple[int, int]'=(512, 288),
                                   mode:str="h264_low"):
        """ 非同步版本的 startRecording
        """
        return await self._call_rpc_async(self.startRecording.__name__,
                                   image_buf=image_buf, image_res=image_res,
                                   save_dirname=save_dirname, cam_idx=cam_idx, mode=mode)

    def startRecording(self, save_dirname:str, cam_idx:int,
                       image_buf:bool=True, image_res:'tuple[int, int]'=(512, 288),
                       mode="h264_low") -> None:
        """開始錄影

        Args:
            save_dirname (str): 儲存的資料夾名稱 ``replay/{$save_dirname}``
            cam_idx (int): 第幾台相機
            image_buf (bool): 是否要將影像同步放到ImageBuffer
            image_res (tuple[int, int]): 影像的解析度
            mode (str, optional): 錄影模式. Defaults to ``h264_low``.

                錄影模式包含以下幾種選擇
                    * ``none``: 不錄影
                    * ``h264_low``: 低畫質
                    * ``h264_high``: 高畫質
                    * ``lossless``: 無損

        Returns:
            None:
        """
        return self._call_rpc_sync(self.startRecording.__name__,
                                   save_dirname=save_dirname,cam_idx=cam_idx,
                                   image_buf=image_buf, image_res=image_res,
                                   mode=mode)

    async def async_stopRecording(self) -> None:
        """停止錄影

        Returns:
            None:
        """
        return await self._call_rpc_async(self.stopRecording.__name__)

    def stopRecording(self):
        return self._call_rpc_sync(self.stopRecording.__name__)

    #def getFile(self, dirname, destination_dir, idx):
    #    res = self._call_rpc_sync(self.getFile.__name__, dirname=dirname)
    #    with open(f"{destination_dir}/CameraReader_{idx}.mp4", "wb") as file:
    #        file.write(res["video"])
    #    with open(f"{destination_dir}/CameraReader_meta_{idx}.csv", "wb") as file:
    #        file.write(res["meta"])

    def startVideoFeeder(self, video_path:str, enable_imgbuf:bool=True):
        return self._call_rpc_sync(self.startVideoFeeder.__name__, video_path=video_path, enable_imgbuf=enable_imgbuf)

    def stopVideoFeeder(self):
        return self._call_rpc_sync(self.stopVideoFeeder.__name__)

    def setIntrinsic(self, ks:np.ndarray, dist:np.ndarray, newcameramtx:np.ndarray) -> None:
        """儲存內參到設定檔

        Args:
            ks (np.ndarray): _description_
            dist (np.ndarray): _description_
            newcameramtx (np.ndarray): _description_

        Returns:
            None: 
        """
        return self._call_rpc_sync(self.setIntrinsic.__name__, ks=ks, dist=dist, newcameramtx=newcameramtx)

    def getIntrinsic(self) -> 'dict[str, list]':
        """取得內參

        Returns:
            dict[str, list]: 內參

            Example::

                {
                    'ks': [[632.5, 0.0, 320.9], [0.0, 630.5, 251.5], [0.0, 0.0, 1.0]],
                    'dist': [[-0.54, 0.48, -0.001, 0.001, -0.404]],
                    'newcameramtx': [[536.21, 0.0, 322.51], [0.0, 577.67, 252.42], [0.0, 0.0, 1.0]]
                }
        """
        return self._call_rpc_sync(self.getIntrinsic.__name__)

    def setExtrinsic(self, poses:np.ndarray, eye:np.ndarray, hmtx:np.ndarray, projection_mat:np.ndarray, extrinsic_mat:np.ndarray):
        """ 設定外參

        Args:
            poses (np.ndarray): _description_
            eye (np.ndarray): _description_
            hmtx (np.ndarray): _description_
            projection_mat (np.ndarray): _description_
            extrinsic_mat (np.ndarray): _description_

        Returns:
            None:
        """
        return self._call_rpc_sync(self.setExtrinsic.__name__, poses=poses, eye=eye, hmtx=hmtx, projection_mat=projection_mat, extrinsic_mat=extrinsic_mat)

    def getExtrinsic(self) -> 'dict[str]':
        """取得外參

        Returns:
            dict[str, list]: 外參

            Example::

                {
                    'poses': [
                        [0.99, 0.007, -0.03],
                        [0.006, -0.458, 0.888],
                        [-0.007, -0.888, -0.458]
                    ],
                    'eye': [[-0.404, -3.20, 2.963]],
                    'hmtx': [
                        [0.08, -0.012, -29.16],
                        [0.0018, -0.107, 43.8],
                        [0.0002, 0.02, 1.0]
                    ],
                    'projection_mat': [
                        [540.98, 277.38, -148.12, 1715.8],
                        [11.3, -63.4, -627.1, 1802.6],
                        [0.01, 0.87, -0.49, 4.39]
                    ],
                    'extrinsic_mat': [
                        [0.99, -0.006, 0.018, 0.558],
                        [0.0129, -0.49, -0.87, 1.2],
                        [0.015, 0.87, -0.49, 4.3]
                    ]
                }
        """
        return self._call_rpc_sync(self.getExtrinsic.__name__)

    def getSnapshot(self):
        """拍照

        Returns:
            numpy.ndarray[numpy.uint8]: 影像 (BGR format)
        """
        return self._call_rpc_sync(self.getSnapshot.__name__)

    def clip(self, replay_path:str, start_time:datetime, duration, save_path):
        res = self._call_rpc_sync(self.clip.__name__, replay_path=replay_path, start_time=start_time, duration=duration)
        with open(save_path, "wb") as file:
            file.write(res)

    async def async_startUdp(self, host:str, port:int):
        return await self._call_rpc_async(self.startUdp.__name__, host=host, port=port)

    def startUdp(self, host:str, port:int):
        """開始UDP串流

        Args:
            host (str): Host
            port (int): Port

        Returns:
            None:
        """
        return self._call_rpc_sync(self.startUdp.__name__, host=host, port=port)

    async def async_stopUdp(self):
        return await self._call_rpc_async(self.stopUdp.__name__)

    def stopUdp(self):
        """停止UDP串流

        Returns:
            None:
        """
        return self._call_rpc_sync(self.stopUdp.__name__)

    def isStreaming(self) -> bool:
        """攝影機是否正在串流(開啟)

        Returns:
            bool: 是否正在串流(開啟)
        """
        return self._call_rpc_sync(self.isStreaming.__name__)

    def getMetricData(self):
        return self._call_rpc_sync("getMetricData")

if __name__ == "__main__":

    from lib.common import ROOTDIR, loadConfig
    from LayerApplication.utils.Mqtt import MqttClient

    cfg_file = f"{ROOTDIR}/config"
    cfg = loadConfig(cfg_file)

    broker_ip = cfg["Project"]["mqtt_broker"]
    broker_port = int(cfg["Project"]["mqtt_port"])
    mqtt_client = MqttClient(broker_ip, broker_port)

    c = RpcCamera("tcam-08324005", mqtt_client.mqttc)
    data = c.getMetricData()
    print(data)
