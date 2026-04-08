import os

from LayerSensing.Pose.PoseMqtt import PoseMqtt
from lib.common import ROOTDIR


class PoseManager:
    def __init__(self, data_handler, distributor):
        self.data_handler = data_handler
        self.distributor = distributor
        self.poseThread = None

    def startPose(self, camera_origin_size: 'tuple[int, int]' = (640, 480), engine_filename: str = 'int8.engine', replay_dirname: str = '', cam_idx: int = 0):
        try:
            if self.poseThread is not None:
                raise Exception('There is another Pose thread is running.')
            engine_path = engine_filename
            if not os.path.isabs(engine_path):
                engine_path = f'{ROOTDIR}/LayerSensing/Pose/weights/{engine_filename}'

            replay_path = f'{ROOTDIR}/replay/{replay_dirname}' if replay_dirname else f'{ROOTDIR}/replay'
            pose_vis_dir = f'{replay_path}/pose'
            os.makedirs(pose_vis_dir, exist_ok=True)

            self.distributor.activate_pose(True)
            self.poseThread = PoseMqtt('Pose', self.data_handler, self.distributor.pose_queue, engine_path, pose_vis_dir)
            self.poseThread.start()
            return {'status': 'ready'}
        except Exception as e:
            return {'status': 'failure', 'message': str(e)}

    def stopPose(self):
        try:
            if self.poseThread is None:
                raise Exception('No pose is running')
            self.distributor.activate_pose(False)
            self.poseThread.stop()
            self.poseThread.join()
            self.poseThread = None
            return {'status': 'stopped'}
        except Exception as e:
            return {'status': 'failure', 'message': str(e)}
