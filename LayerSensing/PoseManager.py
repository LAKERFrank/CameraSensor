import os

from LayerSensing.Pose.PoseMqtt import PoseMqtt
from lib.common import ROOTDIR


class PoseManager:
    def __init__(self, data_handler, distributor):
        self.data_handler = data_handler
        self.distributor = distributor
        self.poseThread = None

    def startPose(self, camera_origin_size: 'tuple[int, int]' = (640, 480), engine_filename: str = 'int8.engine', replay_dirname: str = '', cam_idx: int = 0, source_video_path: str = ''):
        try:
            if self.poseThread is not None:
                raise Exception('There is another Pose thread is running.')
            engine_path = engine_filename
            if not os.path.isabs(engine_path):
                engine_path = f'{ROOTDIR}/LayerSensing/Pose/weights/{engine_filename}'

            if source_video_path:
                replay_path = os.path.dirname(os.path.abspath(source_video_path))
            elif replay_dirname and os.path.isabs(replay_dirname):
                replay_path = replay_dirname
            elif replay_dirname:
                replay_path = f"{ROOTDIR}/replay/{replay_dirname}"
            else:
                replay_path = f"{ROOTDIR}/replay"
            os.makedirs(replay_path, exist_ok=True)
            output_csv = os.path.join(replay_path, f'Pose_{cam_idx}.csv')

            self.distributor.activate_pose(True)
            self.poseThread = PoseMqtt('Pose', self.data_handler, self.distributor.pose_queue, engine_path, output_csv)
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
