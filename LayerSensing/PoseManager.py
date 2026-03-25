import os

from LayerSensing.Pose.PoseMqtt import PoseMqtt
from lib.common import ROOTDIR


class PoseManager:
    DEFAULT_ENGINE_BY_VERSION = {
        'batch1': 'int8.engine',
        'batch3': 'int8_batch3.engine',
    }

    def __init__(self, data_handler, distributor):
        self.data_handler = data_handler
        self.distributor = distributor
        self.poseThread = None

    def startPose(self,
                  camera_origin_size: 'tuple[int, int]' = (640, 480),
                  engine_version: str = 'batch1',
                  engine_filename: str = '',
                  replay_dirname: str = '',
                  cam_idx: int = 0):
        try:
            if self.poseThread is not None:
                raise Exception('There is another Pose thread is running.')

            version = (engine_version or 'batch1').lower()
            if version not in self.DEFAULT_ENGINE_BY_VERSION:
                raise Exception(f'Unsupported pose engine version: {engine_version}. Support: {list(self.DEFAULT_ENGINE_BY_VERSION.keys())}')

            selected_engine = engine_filename or self.DEFAULT_ENGINE_BY_VERSION[version]
            engine_path = selected_engine
            if not os.path.isabs(engine_path):
                engine_path = f'{ROOTDIR}/LayerSensing/Pose/weights/{selected_engine}'

            replay_path = f'{ROOTDIR}/replay/{replay_dirname}' if replay_dirname else f'{ROOTDIR}/replay'
            pose_vis_dir = f'{replay_path}/pose'
            os.makedirs(pose_vis_dir, exist_ok=True)

            self.distributor.activate_pose(True)
            self.poseThread = PoseMqtt('Pose', self.data_handler, self.distributor.pose_queue, engine_path, pose_vis_dir, batch_size=1 if version == 'batch1' else 3)
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
