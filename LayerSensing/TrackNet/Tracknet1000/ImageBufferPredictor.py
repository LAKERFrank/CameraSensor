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

class ImageBufferPredictor:
    def __init__(self, weight: str, image_buffer: ImageBuffer, output_width: int = None,
                 output_height: int = None, mqttc: mqtt.Client = None, data_handler = None,
                 cfg=DEFAULT_CFG, overrides=None, path: str = None, save_pred_images: bool = False, use_nms: bool = True):

        self.args = get_cfg(cfg, overrides)
        self.save_dir = path
        weight = f"{ROOTDIR}/Tracknet1000/weights/best.pt"
        self.model = AutoBackend(weight,
                                 device=select_device(self.args.device, verbose=False),
                                 dnn=self.args.dnn,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=False)
        self.device = self.model.device
        self.args.half = self.model.fp16
        self.model.eval()

        self.output_width = output_width
        self.output_height = output_height
        print(f'[Predictor] Output size: {self.output_width}x{self.output_height}')

        self.mqttc = mqttc
        self.data_handler = data_handler
        self.image_buffer = image_buffer
        self.track_size = 10
        self.imgsz = 640

        self.max_streams = torch.cuda.device_count() * 2
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(self.max_streams)]
        self.event_pool = queue.SimpleQueue()
        self.save_pred_images = save_pred_images
        self.use_nms = use_nms
        self._frame_cache = {}

        for _ in range(512):
            self.event_pool.put(torch.cuda.Event())

        self.stream_idx = 0
        self.infer_q = queue.Queue(maxsize=512)
        self.result_q = queue.Queue(maxsize=512)
        self._stopper = threading.Event()

    def start(self):
        print("[Predictor] Start")
        self.running = True
        self.threads = [
            threading.Thread(target=self._preprocess_loop),
            threading.Thread(target=self._inference_loop),
            threading.Thread(target=self._postprocess_loop),
        ]
        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()  # 阻塞直到所有 thread 結束
        print("[Predictor] All threads finished.")

    def stop(self):
        self.running = False

    def _preprocess_loop(self):
        while self.running:
            try:
                tensor, fids, timestamps = self._preprocess()
                self.stream_idx = (self.stream_idx + 1) % self.max_streams
                try:
                    self.infer_q.put_nowait((tensor, (fids, timestamps), self.stream_idx))
                except queue.Full:
                    LOGGER.warning("[Predictor] infer_q 已滿，無法放入新資料")
                if not self.running:
                    self.infer_q.put(None)
            except Exception as e:
                LOGGER.warning(f"Preprocess loop error: {e}")

    def _preprocess(self) -> Tuple[torch.Tensor, List[int], List[float]]:
        frames, fids, timestamps = [], [], []
        while len(frames) < self.track_size:
            frame = self.image_buffer.pop(True)
            if frame.is_eos:
                self.stop()
                self.data_handler.publish("tracknet", json.dumps({"linear": [], "EOF": True}))
                break

            # 目前傳入的尺寸不精確，在此額外處理
            self.output_height, self.output_width = frame.image.shape[:2]

            if self.save_pred_images:
                self._frame_cache[frame.index] = frame.image.copy()
            img = frame.image.astype(np.float32)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self.pad_to_square(img)
            img = cv2.resize(img, dsize=(self.imgsz, self.imgsz), interpolation=cv2.INTER_CUBIC)
            frames.append(np.expand_dims(img, axis=0))
            fids.append(frame.index)
            timestamps.append(frame.monotonic_timestamp)

        if len(frames) < self.track_size:
            for _ in range(self.track_size - len(frames)):
                img = np.zeros((self.imgsz, self.imgsz), dtype=np.float32)
                frames.append(np.expand_dims(img, axis=0))
                fids.append(-1)
                timestamps.append(-1)

        img = np.concatenate(frames, 0)
        # pin_memory() 可能會會消耗過多 CPU，效益不明顯
        return torch.from_numpy(img).contiguous(), fids, timestamps
        # return torch.from_numpy(img).contiguous().pin_memory(), fids, timestamps

    def _inference_loop(self):
        while self.running:
            try:
                item = self.infer_q.get()
                if item is None:
                    self.result_q.put(None)
                    break

                tensor, meta, stream_id = item
                stream = self.streams[stream_id]
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
                try:
                    self.result_q.put_nowait((event, output, meta, stream))
                except queue.Full:
                    LOGGER.warning("[Predictor] result_q 已滿，無法放入新資料")
            except Exception as e:
                LOGGER.warning(f"Inference loop error: {e}")

    def _postprocess_loop(self):
        while self.running:
            try:
                item = self.result_q.get()
                if item is None:
                    break
                event, output, meta, stream = item
                if event.query():
                    self._postprocess(output, meta)
                    self.event_pool.put(event)
                else:
                    # 沒完成的重新放回，但避免 busy loop
                    try:
                        self.result_q.put_nowait((event, output, meta, stream))
                    except queue.Full:
                        LOGGER.warning("[Predictor] result_q 已滿，無法放入新資料")
                    time.sleep(0.01)
            except Exception as e:
                LOGGER.warning(f"Postprocess loop error: {e}")
                raise e


    def pad_to_square(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        size = max(h, w)
        padded = np.zeros((size, size), dtype=img.dtype)
        padded[:h, :w] = img
        return padded

    def _postprocess(self, output_tensor: torch.Tensor, meta:Tuple[List[int], List[float]]):
        fids, timestamps = meta

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
            pred_local = []
            if self.use_nms:
                nms_preds = non_max_suppression(p_conf, p_cell_x, p_cell_y, conf_threshold=conf_threshold, dis_tolerance=20)

                # 取出 nms 的結果
                for pred in nms_preds:
                    max_x, max_y, max_conf = pred
                    pred_x = max_x*stride + (center*stride-p_cell_x[int(max_y)][int(max_x)][0]+p_cell_x[int(max_y)][int(max_x)][1])
                    pred_y = max_y*stride + (center*stride-p_cell_y[int(max_y)][int(max_x)][0]+p_cell_y[int(max_y)][int(max_x)][1])

                    if fid != -1:
                        real_result = self.convert_coord_to_original_ratio(coord=(pred_x.item(), pred_y.item(), max_conf))
                        frame_preds.append(real_result)
                        metadata.append((fid, timestamp))
                        if self.save_pred_images:
                            pred_local.append(real_result)
            else:
                p_conf_masked = p_conf * (p_conf >= conf_threshold).float()
                max_position = torch.argmax(p_conf_masked)
                # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
                max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
                max_conf = p_conf[max_y, max_x].item()
                if max_conf < conf_threshold:
                    continue

                pred_x = max_x*stride + (center*stride-p_cell_x[max_y][max_x][0]+p_cell_x[max_y][max_x][1])
                pred_y = max_y*stride + (center*stride-p_cell_y[max_y][max_x][0]+p_cell_y[max_y][max_x][1])
                if fid != -1:
                    real_result = self.convert_coord_to_original_ratio(coord=(pred_x.item(), pred_y.item(), max_conf))
                    frame_preds.append(real_result)
                    metadata.append((fid, timestamp))
                    if self.save_pred_images:
                        pred_local.append(real_result)
            
            if self.save_pred_images:
                img = self._frame_cache.get(fid)
                self.save_image_with_points(pred_local, img, f"{self.save_dir}/{fid}.png")

        result = (frame_preds, metadata)
        if self.mqttc is not None:
            self._publishPoints((frame_preds, metadata) if self.use_nms else (frame_preds[:1], metadata[:1]))
        # print("[Result] output shape:", len(frame_preds), "fid", fid, "timestamp", timestamp, "endTime", time.monotonic())
        return result

    def convert_coord_to_original_ratio(
        self,
        coord: Tuple[float, float, float],
        model_input_size: int = 640
    ) -> Tuple[float, float, float]:
        """
        將單一 (x, y, conf) 從 640x640 padding 空間還原到原始影像座標比例。
        因為 pad_to_square 是左上角對齊，還原時不需要減去 pad_x, pad_y。

        Args:
            coord: (x, y, conf)
            model_input_size: 模型輸入大小(預設 640)
            
        Returns:
            (x_orig, y_orig, conf)
        """
        x, y, conf = coord

        # 取得 padded 空間大小
        size = max(self.output_width, self.output_height)
        scale = size / model_input_size

        # 直接放大，因為 pad_x/pad_y = 0
        x_orig = x * scale
        y_orig = y * scale

        # 限制在原圖範圍內
        x_orig = max(0, min(self.output_width - 1, x_orig))
        y_orig = max(0, min(self.output_height - 1, y_orig))

        return (x_orig, y_orig, conf)



    def _publishPoints(self, resultItems):
        (frame_preds, metadata) = resultItems
        points = []
        for i in range(len(frame_preds)):
            (output_x, output_y, output_conf) = frame_preds[i]
            (fid, timestamp) = metadata[i]

            points.append(Point(
                fid=fid,
                timestamp=timestamp,
                visibility=output_conf>=0.5,
                x=output_x,
                y=output_y,
                ))
        payload = {"linear": [p.toJson() for p in points]}
        self.data_handler.publish("tracknet", json.dumps(payload))
    def save_image_with_points(self, points: List[Tuple[float, float, float]], image: np.ndarray, save_path: str):
        """
        將點集 (x, y, conf) 繪製到影像上，並保存到指定路徑。
        
        Args:
            points (List[Tuple[float, float, float]]): [(x, y, conf), ...]
            image (np.ndarray): BGR 或灰階影像 (H, W, C) 或 (H, W)。
            save_path (str): 儲存影像的完整路徑。
        """
        # 複製影像避免修改到原始
        img_vis = image.copy()
        if img_vis.ndim == 2:
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

        for x, y, conf in points:
            if conf < 0.5:
                color = (0, 0, 255)  # 紅色，低信心
            else:
                color = (0, 255, 0)  # 綠色，正常

            center = (int(x), int(y))
            print(f"[Predictor] Drawing point at {center} with confidence {conf:.2f}")
            cv2.circle(img_vis, center, radius=3, color=color, thickness=-1)

        cv2.imwrite(save_path, img_vis)