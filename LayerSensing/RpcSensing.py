from lib.Rpc import RemoteProcedureCall

class RpcSensing(RemoteProcedureCall):
    def __init__(self, device_name, mqtt_client):
        super().__init__(device_name, "SensingLayer", mqtt_client)
        self._datafeeder_prefix = "TrackNet"
        
    def startTrackNet(self, camera_origin_size:'tuple[int, int]',
                      tracknet_ver:str, weights_filename:str,
                      replay_dirname:str, cam_idx: int):
        """Start Tracknet thread

        Args:
            camera_origin_size (tuple[int, int]): 相機原始解析度 (Tracknet會回推)
            tracknet_ver (str): Tracknet版本 ("tracknet_v2" or "tracknet_1000")
            weights_filename (str): 模型檔案名稱
            replay_dirname (str): 儲存的資料夾名稱 
            cam_idx (int): 相機編號

        Returns:
            dict: 狀態
        """

        return self._call_rpc_sync("TrackNet/start",
                                   camera_origin_size = camera_origin_size,
                                   tracknet_ver=tracknet_ver,
                                   weights_filename=weights_filename,
                                   replay_dirname=replay_dirname,
                                   cam_idx = cam_idx)

    def stopTrackNet(self):
        """Stop Tracknet thread

        Returns:
            dict: 狀態
        """
        return self._call_rpc_sync("TrackNet/stop", timeout=1000)

    def startPose(
        self,
        engine_filename: str,
        cam_idx: int,
        *,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_det: int = 100,
    ):
        """Start Pose thread with a TensorRT engine.

        Args:
            engine_filename (str): Engine 檔案名稱或絕對路徑
            cam_idx (int): 相機編號
            input_size (int, optional): 模型輸入尺寸. Defaults to 640.
            conf_threshold (float, optional): 置信度閾值. Defaults to 0.25.
            iou_threshold (float, optional): IoU 閾值. Defaults to 0.65.
            max_det (int, optional): 最大偵測數量. Defaults to 100.

        Returns:
            dict: 狀態
        """

        return self._call_rpc_sync(
            "Pose/start",
            engine_filename=engine_filename,
            cam_idx=cam_idx,
            input_size=input_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
        )

    def stopPose(self):
        """Stop Pose thread

        Returns:
            dict: 狀態
        """

        return self._call_rpc_sync("Pose/stop", timeout=1000)

    def startDatafeeder(
        self,
        filepath,
        metapath=None,
        posepath=None,
        *,
        pose_playback_speed: float = 1.0,
    ):
        payload = {"filepath": filepath}
        if metapath is not None:
            payload["metapath"] = metapath
        if posepath is not None:
            payload["posepath"] = posepath
        if pose_playback_speed != 1.0:
            payload["pose_playback_speed"] = pose_playback_speed
        prefix = "Pose" if posepath is not None else "TrackNet"
        self._datafeeder_prefix = prefix
        return self._call_rpc_sync(f"{prefix}/startDatafeeder", **payload)

    def stopDatafeeder(self):
        prefix = getattr(self, "_datafeeder_prefix", "TrackNet")
        return self._call_rpc_sync(f"{prefix}/stopDatafeeder")
