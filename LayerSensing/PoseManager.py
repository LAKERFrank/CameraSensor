import os
import queue

from LayerCamera.CameraSystemC.recorder_module import Frame
from LayerSensing.FrameDistributor import CONSUMER_POSE, SharedFrameState
from LayerSensing.Pose.PoseMqtt import PoseMqtt
from lib.common import ROOTDIR


class PoseManager:
    DEFAULT_ENGINE_BY_VERSION = {
        'batch1': 'int8.engine',
        'batch3': 'int8_batch3.engine',
        'batch10': 'int8_batch10.engine',
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
            pose_json_path = f'{pose_vis_dir}/Pose_{cam_idx}.jsonl'

            self.distributor.activate_pose(True)
            batch_size = {'batch1': 1, 'batch3': 3, 'batch10': 10}[version]
            self.poseThread = PoseMqtt('Pose', self.data_handler, self.distributor.pose_queue, batch_size=batch_size, engine_path=engine_path, output_json=pose_json_path)
            self.poseThread.start()
            return {'status': 'ready'}
        except Exception as e:
            return {'status': 'failure', 'message': str(e)}

    def stopPose(self, wait_for_eos=False):
        try:
            if self.poseThread is None:
                raise Exception('No pose is running')

            if not wait_for_eos:
                frame = Frame()
                frame.is_eos = True
                eos_state = SharedFrameState(frame=frame, need_mask=CONSUMER_POSE)
                try:
                    self.distributor.pose_queue.push_state(eos_state)
                except queue.Full:
                    dropped = self.distributor.pose_queue.pop_nowait_state()
                    self.distributor.release(dropped, CONSUMER_POSE)
                    self.distributor.pose_drop_count += 1
                    self.distributor.pose_queue.push_state(eos_state)

            self.poseThread.join()
            self.poseThread = None
            self.distributor.activate_pose(False)
            return {'status': 'stopped ' + ('(EOS reached)' if wait_for_eos else '(Force stop)')}
        except Exception as e:
            return {'status': 'failure', 'message': str(e)}
