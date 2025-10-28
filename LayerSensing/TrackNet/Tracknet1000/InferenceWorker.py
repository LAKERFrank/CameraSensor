import json
from pathlib import Path
import threading
import time
import queue
from typing import Tuple, List

import cv2
import numpy as np
import torch
from lib.point import Point
import paho.mqtt.client as mqtt

from LayerCamera.CameraSystemC.recorder_module import ImageBuffer
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device
import os
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)

inference_worker_instance = None
_instance_lock = threading.Lock()

def get_inference_worker(weight: str, output_width: int = None,
                 output_height: int = None, mqttc: mqtt.Client = None, output_topic: str = None,
                 cfg=DEFAULT_CFG, overrides=None):
    global inference_worker_instance
    if inference_worker_instance is None:
        with _instance_lock:
            if inference_worker_instance is None:
                inference_worker_instance = InferenceWorker(weight, output_width, output_height, mqttc, output_topic, cfg, overrides)
    return inference_worker_instance

# inference_worker_instance = None

# def get_inference_worker(weight: str, output_width: int = None,
#                  output_height: int = None, mqttc: mqtt.Client = None, output_topic: str = None,
#                  cfg=DEFAULT_CFG, overrides=None):
#     global inference_worker_instance
#     if inference_worker_instance is None:
#         inference_worker_instance = InferenceWorker(weight, output_width, output_height, mqttc, output_topic, cfg, overrides)
#     return inference_worker_instance

def non_max_suppression(pred_conf, pred_x, pred_y, conf_threshold=0.5, dis_tolerance=10, stride=8):
        """
        Apply non-maximum suppression (NMS) to filter out overlapping balls based on distance.
        
        Args:
        - pred_conf (torch.Tensor): 80x80 confidence tensor.
        - pred_x (torch.Tensor): 80x80 x-coordinate tensor.
        - pred_y (torch.Tensor): 80x80 y-coordinate tensor.
        - conf_threshold (float): Confidence threshold.
        - dis_tolerance (float): Minimum allowable distance between two detections.
        
        Returns:
        - keep (list): List of (x, y, confidence) tuples for retained detections.
        """
        # 1. 篩選置信度 >= conf_threshold 的 cell
        mask = pred_conf >= conf_threshold
        conf_values = pred_conf[mask]
        indices = torch.nonzero(mask)

        # 2. 依照置信度降序排序
        sorted_indices = torch.argsort(conf_values, descending=True)
        sorted_conf_values = conf_values[sorted_indices]
        sorted_positions = indices[sorted_indices]

        # 3. 初始化結果列表
        keep = []
        result = []
        center = stride/2
        # 4. 應用距離檢查的 NMS
        for i in range(len(sorted_conf_values)):

            y_coordinates, x_coordinates = sorted_positions[i].tolist()
            x1 = center*stride - pred_x[y_coordinates][x_coordinates][0]+pred_x[y_coordinates][x_coordinates][1]
            y1 = center*stride - pred_y[y_coordinates][x_coordinates][0]+pred_y[y_coordinates][x_coordinates][1]
            conf = sorted_conf_values[i].item()

            x_coordinates *= stride
            y_coordinates *= stride
            current_x = x_coordinates+x1
            current_y = y_coordinates+y1

            # 只保留與所有已保留框距離大於 dis_tolerance 的框
            is_far_enough = True
            for x2, y2, _ in keep:
                distance = torch.sqrt((current_x - x2) ** 2 + (current_y - y2) ** 2)
                if distance < dis_tolerance:
                    is_far_enough = False
                    break

            if is_far_enough:
                keep.append((current_x, current_y, conf))
                result.append((x_coordinates/stride, y_coordinates/stride, conf))

        return result

# 請使用 get_inference_worker() 來獲取 InferenceWorker 實例
class InferenceWorker:
    def __init__(self, weight: str, output_width: int = None,
                 output_height: int = None, mqttc: mqtt.Client = None, output_topic: str = None,
                 cfg=DEFAULT_CFG, overrides=None):

        self.args = get_cfg(cfg, overrides)
        self.args.half = True
        weight = f"{ROOTDIR}/Tracknet1000/weights/best.pt"
        self.model = AutoBackend(weight,
                                 device=select_device(self.args.device, verbose=False),
                                 dnn=self.args.dnn,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=False)
        self.device = self.model.device
        print(f"Using device: {self.device}")
        self.args.half = self.model.fp16
        self.model.eval()

        self.output_width = output_width
        self.output_height = output_height
        self.mqttc = mqttc
        self.output_topic = output_topic
        self.track_size = 10
        self.imgsz = 640

        self.max_streams = torch.cuda.device_count() * 2
        print(f"Max streams: {self.max_streams}")
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(self.max_streams)]
        self.event_pool = queue.SimpleQueue()

        for _ in range(2000):
            self.event_pool.put(torch.cuda.Event())

        self.stream_idx = 0
        self.infer_q = queue.Queue(maxsize=512)
        self.result_q = queue.Queue(maxsize=512)
    
    def start(self):
        print("[InferenceWorker] Start", time.monotonic())
        self.running = True
        self.threads = [
            threading.Thread(target=self._inference_loop),
            threading.Thread(target=self._postprocess_loop),
        ]
        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()  # 阻塞直到所有 thread 結束
        print("[InferenceWorker] All threads finished.", time.monotonic())

    def stop(self):
        self.running = False

    def _inference_loop(self):
        while self.running:
            try:
                item = self.infer_q.get(timeout=0.1)
                if item is None:
                    self.result_q.put(None)
                    break

                tensor, meta = item
                self.stream_idx = (self.stream_idx + 1) % self.max_streams
                stream = self.streams[self.stream_idx]
                event = self.event_pool.get()

                with torch.cuda.stream(stream):
                    input_gpu = tensor.to(self.device, non_blocking=True)
                    median = input_gpu.median(dim=0).values
                    input_gpu.sub_(median).clamp_(0, 255).div_(255.0)
                    input_gpu = input_gpu.unsqueeze(0)
                    if self.model.fp16:
                        input_gpu = input_gpu.half()
                    with torch.no_grad():
                        output = self.model(input_gpu)
                    event.record(stream)

                self.result_q.put((event, output, meta, stream))
            except queue.Empty:
                continue
            except Exception as e:
                LOGGER.warning(f"Inference loop error: {e}")

    def _postprocess_loop(self):
        while self.running:
            try:
                item = self.result_q.get(timeout=0.1)
                if item is None:
                    break
                event, output, meta, stream = item
                if event.query():
                    self._postprocess(output, meta)
                    self.event_pool.put(event)
                else:
                    # 沒完成的重新放回，但避免 busy loop
                    self.result_q.put((event, output, meta, stream))
                    time.sleep(0.001)
            except queue.Empty:
                time.sleep(0.001)
            except Exception as e:
                LOGGER.warning(f"Postprocess loop error: {e}")
                raise e

    def _postprocess(self, output_tensor: torch.Tensor, meta:Tuple[List[int], List[float]]):
        fids, timestamps = meta

        use_nms = True
        conf_threshold = 0.5
        nc = 1
        feat_no = 8
        cell_num = 80
        stride = 8

        feats = output_tensor[0][0]
        pred_distri, pred_probs = feats.view(feat_no + nc, -1).split(
            (feat_no, nc), 0)
        
        pred_probs = pred_probs.permute(1, 0)
        pred_pos = pred_distri.permute(1, 0)

        each_probs = pred_probs.view(10, cell_num, cell_num)
        each_pos_x, each_pos_y, each_pos_nx, each_pos_ny = pred_pos.view(10, cell_num, cell_num, feat_no).split([2, 2, 2, 2], dim=3)

        frame_preds = []
        metadata = []
        for frame_idx in range(10):
            p_cell_x = each_pos_x[frame_idx]
            p_cell_y = each_pos_y[frame_idx]
            p_cell_nx = each_pos_nx[frame_idx]
            p_cell_ny = each_pos_ny[frame_idx]
            fid = fids[frame_idx]
            timestamp = timestamps[frame_idx]
            center = 0.5

            # 獲取當前圖片的 conf
            p_conf = each_probs[frame_idx]

            if use_nms:
                nms_preds = non_max_suppression(p_conf, p_cell_x, p_cell_y, conf_threshold=conf_threshold, dis_tolerance=20)

                # 取出 nms 的結果
                for pred in nms_preds:
                    max_x, max_y, max_conf = pred
                    pred_x = max_x*stride + (center*stride-p_cell_x[int(max_y)][int(max_x)][0]+p_cell_x[int(max_y)][int(max_x)][1])
                    pred_y = max_y*stride + (center*stride-p_cell_y[int(max_y)][int(max_x)][0]+p_cell_y[int(max_y)][int(max_x)][1])

                    if fid != -1:
                        frame_preds.append((pred_x, pred_y, max_conf))
                        metadata.append((fid, timestamp))
            else:
                p_conf_masked = p_conf * (p_conf >= conf_threshold).float()
                max_position = torch.argmax(p_conf_masked)
                # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
                max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
                max_conf = p_conf[max_y, max_x].item()

                pred_x = max_x*stride + (center*stride-p_cell_x[max_y][max_x][0]+p_cell_x[max_y][max_x][1])
                pred_y = max_y*stride + (center*stride-p_cell_y[max_y][max_x][0]+p_cell_y[max_y][max_x][1])
                if fid != -1:
                    frame_preds.append((pred_x, pred_y, max_conf))
                    metadata.append((fid, timestamp))
        result = (frame_preds, metadata)
        if self.mqttc is not None:
            self._publishPoints((frame_preds, metadata) if use_nms else (frame_preds[:1], metadata[:1]))
        # print("[Result] output shape:", len(frame_preds), "fid", fid, "timestamp", timestamp, "endTime", time.monotonic())
        return result

    def _publishPoints(self, resultItems):
        (frame_preds, metadata) = resultItems
        points = []
        for i in range(len(frame_preds)):
            (output_x, output_y, output_conf) = frame_preds[i]
            (fid, timestamp) = metadata[i]
            output_x = output_x.item()
            output_y = output_y.item()

            points.append(Point(
                fid=fid,
                timestamp=timestamp,
                visibility=output_conf>=0.5,
                x=output_x,
                y=output_y,
                ))
        payload = {"linear": [p.toJson() for p in points]}
        self.mqttc.publish(self.output_topic, json.dumps(payload))
