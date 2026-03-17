from lib.Rpc import RemoteProcedureCall

class RpcSensing(RemoteProcedureCall):
    def __init__(self, device_name, mqtt_client):
        super().__init__(device_name, "SensingLayer", mqtt_client)
        
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

    def startDatafeeder(self, filepath):
        return self._call_rpc_sync("TrackNet/startDatafeeder", filepath=filepath)

    def stopDatafeeder(self):
        return self._call_rpc_sync("TrackNet/stopDatafeeder")

    def startPose(self, camera_origin_size:'tuple[int, int]' = (640, 480),
                  engine_filename: str = "int8.engine",
                  replay_dirname: str = "",
                  cam_idx: int = 0,
                  source_video_path: str = ""):
        """Start Pose thread

        Args:
            camera_origin_size (tuple[int, int]): 相機原始解析度 (目前保留欄位，便於和 TrackNet 對齊)
            engine_filename (str): TensorRT engine filename or absolute path.
            replay_dirname (str): 儲存資料夾名稱 (目前保留欄位，便於和 TrackNet 對齊)
            cam_idx (int): 相機編號 (目前保留欄位，便於和 TrackNet 對齊)
            source_video_path (str): 來源影片路徑；提供時 CSV 會存到影片所在資料夾。

        Returns:
            dict: 狀態
        """
        return self._call_rpc_sync("Pose/start",
                                   camera_origin_size=camera_origin_size,
                                   engine_filename=engine_filename,
                                   replay_dirname=replay_dirname,
                                   cam_idx=cam_idx,
                                   source_video_path=source_video_path)

    def stopPose(self):
        """Stop Pose thread

        Returns:
            dict: 狀態
        """
        return self._call_rpc_sync("Pose/stop", timeout=1000)
