"""
EventDetector : To recognize the event (Hit, Land) by trajectory of 3D-Model
"""
import logging
import paho.mqtt.client as mqtt
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import threading
import csv
import pandas as pd

from datetime import datetime
from dataclasses import dataclass, field, replace
from typing import List, Tuple, Optional, Union
from enum import Enum, auto
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from math import isfinite

'''
Our common function
'''
from lib.common import ROOTDIR, loadConfig, loadNodeConfig
from lib.inspector import sendPerformance, sendNodeStateMsg
from lib.point import Point, sendPoints
from lib.writer import CSVWriter

from LayerContent.isPhysics import trajectory_segment, optimize_velocity, physics_predict3d_v2
from LayerContent.CES.smooth import detectBallType, detectBallTypeDT

EPS = 1e-9

def _angle_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    c = np.clip(np.dot(a, b)/(na*nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def angle_xy_deg(prev_v, v, signed=False):
    """
    水平平面 (x,y) 的夾角。
    - signed=False：回傳 0~180°
    - signed=True ：回傳 (-180, 180]，以 +Z 為正向（逆時針為正）
    """
    ax, ay = float(prev_v[0]), float(prev_v[1])
    bx, by = float(v[0]),      float(v[1])

    a2 = np.array([ax, ay], dtype=float)
    b2 = np.array([bx, by], dtype=float)

    if not signed:
        return _angle_deg(a2, b2)

    # 有號角度：atan2( det, dot )，det>0 代表逆時針（+Z）
    dot = ax*bx + ay*by
    det = ax*by - ay*bx  # 等同 2D cross 的 z 分量
    if np.hypot(ax, ay) < EPS or np.hypot(bx, by) < EPS:
        return 0.0
    return float(np.degrees(np.arctan2(det, dot)))  # (-180, 180]

def angle_yz_deg(prev_v, v, signed=False):
    """
    垂直平面 (y,z) 的夾角。
    - signed=False：回傳 0~180°
    - signed=True ：回傳 (-180, 180]，以 +X 為正向（從 y 指向 z 逆時針為正）
    """
    ay, az = float(prev_v[1]), float(prev_v[2])
    by, bz = float(v[1]),      float(v[2])

    a2 = np.array([ay, az], dtype=float)
    b2 = np.array([by, bz], dtype=float)

    if not signed:
        return _angle_deg(a2, b2)

    # 有號角度：把 (y,z) 視為平面座標，正向定為 +X
    dot = ay*by + az*bz
    det = ay*bz - az*by  # 相當於 3D 中 cross(a,b) 的 +X 分量
    if np.hypot(ay, az) < EPS or np.hypot(by, bz) < EPS:
        return 0.0
    return float(np.degrees(np.arctan2(det, dot)))  # (-180, 180]

def _speed_ratio(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if min(na, nb) < 1e-6: return 1.0
    return max(na, nb) / max(1e-6, min(na, nb))


class DetectorState(Enum):
    IDLE = auto()
    AWAIT_SERVE = auto()
    RALLY = auto()
    AWAIT_DEAD = auto()

@dataclass
class DetectorConfig():
    fps: int = 120

    carryover_max_gap = 0.5
    
    inlier_threshold = 0.3
    inlier_ratio = 0.7
    
    sequence_length: int = 20
    flight_speed: float = 1.0
    height_threshold: float = 2.0
    shot_length: int = 20

    # serve_height: Tuple[float, float] = (0.5, 2.5)
    serve_height: Tuple[float, float] = (0.5, 3.0)
    # serve_region: dict = field(default_factory=lambda: {'x': (-2.0, 2.0),
    #                                                     'y': (-5.0, -1.0)})
    serve_region: dict = field(default_factory=lambda: {'x': (-2.0, 2.0),
                                                        'y': (-5.0, -0.5)})
    min_points_for_serve: int = 10
    min_serve_dy: float = 1.5

    # 新增：各狀態的最大延遲（以「幀」為單位，便於跟 fps 對齊）
    max_delay_idle_frames: int = 360
    max_delay_await_serve_frames: int = 360
    max_delay_rally_frames: int = 360
    max_delay_await_dead_frames: int = 240

    def max_delay_by_state(self, state: DetectorState) -> float:
        frames = {
            DetectorState.RALLY: self.max_delay_rally_frames,
            DetectorState.AWAIT_DEAD: self.max_delay_await_dead_frames,
            DetectorState.AWAIT_SERVE: self.max_delay_await_serve_frames,
            DetectorState.IDLE: self.max_delay_idle_frames,
        }.get(state, self.max_delay_idle_frames)
        return frames / self.fps
    
    @property
    def max_delay(self) -> float:
        return 360 * (1 / self.fps)


class CSVWriter():
    def __init__(self, filename):
        self.filename = filename
        self.csvfile = open(filename, 'w', newline='')
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow(['Type', 'fid', 'event', 'timestamp', 'position', 'start_fid', 'start_timestamp', 'start_position', 'end_fid', 'end_timestamp', 'end_position', 'speed', 'ball_type', 'publish_time'])

    def close(self):
        self.csvfile.flush()
        self.csvfile.close()

    def writePublish(self, type, data, publish_time):
        if type == 'Event':
            self.writer.writerow(['Event', data['fid'], data['event'], data['timestamp'], json.dumps(data['position']), '', '', '', '', '', '', '', '', publish_time])
        elif type == 'Segment':
            self.writer.writerow(['Segment', '', '', '', '', data['start_fid'], data['start_timestamp'], json.dumps(data['start_position']), data['end_fid'], data['end_timestamp'], json.dumps(data['end_position']), json.dumps(data['speed']), data['ball_type'], publish_time])

        self.csvfile.flush()

class Publisher:
    def publish_event(self, payload):
        raise NotImplementedError("This method should be overridden by subclasses.")
    def publish_segment(self, patload):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class MQTTPublisher(Publisher):
    def __init__(self, det, data_handler, client, topic_root: str, csv_writer, fps):
        self.det = det
        self.data_handler = data_handler
        self.client = client
        self.topic_root = topic_root.rstrip("/")
        self.writer = csv_writer
        self.fps = fps

    def _pub(self, suffix, payload, label):
        if label == "Event" or label == "Segment":
            self.data_handler.publish(suffix, payload)
            t = datetime.now().time().isoformat(timespec="milliseconds")
            self.writer.writePublish(label, payload, t)
            logging.info(f"Publish [{suffix}]: {payload} ")
        else:
            topic = f"{self.topic_root}/{suffix}"
            self.client.publish(topic, json.dumps(payload))
            
    def publish_event(self, event_type, point):
        payload = self.make_event_payload(event_type, point)
        self._pub("event", payload, "Event")

        json_event = {
            "type": "event",
            "event_type": {1: "hit", 2: "serve", 3: "dead"}.get(event_type, "unknown"),
            "fid": payload["fid"],
            "timestamp": payload["timestamp"],
            "position": payload["position"]
        }

        ts = point.timestamp
        if event_type == 2: # serve
            if self.det.current_phase is not None:
                self.det._end_current_phase(ts)
            self.det._start_new_phase("rally", ts)
            self.det.current_description.append(json_event)
        elif event_type == 3: # dead
            self.det.current_description.append(json_event)
            self.det._end_current_phase(ts)
            self.det._start_new_phase("rest", ts)
        else:
            self.det.current_description.append(json_event)
    
    def publish_segment(self, points, extend_to_ground=False):
        pred_traj = None
        payload, _pred_traj = self.make_segment_payload(points, extend_to_ground)
        self._pub("segment", payload, "Segment")

        json_segment = {
            "type": "segment",
            "start_fid": payload["start_fid"],
            "start_timestamp": payload["start_timestamp"],
            "start_position": payload["start_position"],
            "end_fid": payload["end_fid"],
            "end_timestamp": payload["end_timestamp"],
            "end_position": payload["end_position"],
            "speed": payload["speed"],
            "ball_type": payload["ball_type"]
        }
        self.det.current_description.append(json_segment)

        if extend_to_ground and _pred_traj is not None and len(_pred_traj) > 0:
            pred_traj = _pred_traj
            logging.debug(f"[LandingDebug] t:{_pred_traj[-1][3]}, pos:({_pred_traj[-1][0]}, {_pred_traj[-1][1]}, {_pred_traj[-1][2]})")
            # 取最後一點當「預測落地點」(touch_ground_cut=True → 最後點就是地面)
            xL, yL, zL, tL = map(float, _pred_traj[-1])
            landing_debug = {
                'type': 'segment_landing',
                'start_fid': payload.get('start_fid'),
                'end_fid': payload.get('end_fid'),
                'pred_linear': self._traj2linear(_pred_traj),  # 給前端畫虛線/點
                'landing': {
                    'pos': {'x': xL, 'y': yL, 'z': zL},
                    'timestamp': round(tL, 3)
                }
            }
            self._pub("Landing/Debug", landing_debug, "LandingDebug")
        return pred_traj

    def make_event_payload(self, event_type, point):
        if not event_type or not point:
            return {}
        if event_type == 1 or event_type == 2 or event_type == 3:
            position = [point.x, point.y, point.z]
        elif event_type == 4 or event_type == 5:
            position = [-1, -1, -1]
        payload = {'fid': point.fid, 'event': event_type, 'timestamp': round(point.timestamp, 3), 'position': [round(pos, 3) for pos in position]}
        return payload

    def make_segment_payload(self, points, extend_to_ground):
        _pred_traj = None
        pred_points = None
        if not points:
            return {}, _pred_traj
        
        start = points[0]
        end = points[-1]
        starting_point = [start.x, start.y, start.z, start.timestamp]
        flight_time = end.timestamp - start.timestamp
        initial_V = self.estimate_initial_v(points, g=9.81, M=6)
        logging.debug(f"[EstimateV]: {initial_V}")
        optimized_V, loss = optimize_velocity(points, starting_point, self.fps, flight_time, np.array(initial_V))
        if optimized_V is not None:
            logging.debug(f"{optimized_V}, {np.linalg.norm(optimized_V)*3.6:.3f} km/h")
            optimized_V = list(optimized_V)
            if extend_to_ground:
                _pred_traj = physics_predict3d_v2(starting_point, np.array(optimized_V), self.fps, 10.0, touch_ground_cut=True, alpha=0.242)
            else:
                seg_dur = max(end.timestamp - start.timestamp, 1.0 / self.fps)
                _pred_traj = self._predict_traj_until_pos(starting_point, np.array(optimized_V), self.fps, end, alpha=0.242, base_dur=seg_dur, extra_time=0.2, max_dur=2.0, dist_tol=0.50)

            if _pred_traj is not None and len(_pred_traj) > 0:
                try:
                    pred_points = self._traj_numpy_to_points(_pred_traj, base_fid=start.fid, visibility=1)
                    self.publish_balltype(pred_points)
                except Exception as e:
                    logging.warning(f"Wrap _pred_traj to Point failed, fallback to raw points. err={e}")
        else:
            logging.debug(f"None, [0, 0, 0]")
            optimized_V = [0, 0, 0]
        ball_type = detectBallTypeDT(points, pred_points, optimized_V)
        payload = {'start_fid': start.fid, 'start_timestamp': round(start.timestamp, 3), 'start_position': [round(start.x, 3), round(start.y, 3), round(start.z, 3)], 
                   'end_fid': end.fid, 'end_timestamp': round(end.timestamp, 3), 'end_position': [round(end.x, 3), round(end.y, 3), round(end.z, 3)],
                   'speed': [round(v, 3) for v in optimized_V], 'ball_type': ball_type}
        return payload, _pred_traj
    
    def estimate_initial_v(self, points, g=9.81, M=6):
        if len(points) < 2:
            return np.array([0., 0., 0.])

        M = min(M, len(points))
        p0  = points[0]
        t0  = p0.timestamp
        ts  = np.array([p.timestamp - t0 for p in points[:M]], dtype=float)

        xs  = np.array([p.x for p in points[:M]], dtype=float)
        ys  = np.array([p.y for p in points[:M]], dtype=float)
        zs  = np.array([p.z for p in points[:M]], dtype=float)

        # 線性擬合 x(t), y(t)：[a, b] with x ~ a*t + b
        # a 就是 v0
        if np.allclose(ts, ts[0]):  # 時間戳若無變化就退化處理
            dt = (points[1].timestamp - points[0].timestamp) or 1e-3
            vx0 = (points[min(2, len(points)-1)].x - p0.x) / (min(2, len(points)-1) * dt)
            vy0 = (points[min(2, len(points)-1)].y - p0.y) / (min(2, len(points)-1) * dt)
            # z 做重力補償差分
            vz0 = (points[min(2, len(points)-1)].z - p0.z + 0.5*g*(min(2, len(points)-1)*dt)**2) / (min(2, len(points)-1)*dt)
            return np.array([vx0, vy0, vz0])

        ax, bx = np.polyfit(ts, xs, 1)
        ay, by = np.polyfit(ts, ys, 1)

        # z：方法A：直接二次 fit，z(t) ≈ a2*t^2 + a1*t + a0，理論上 a2 ≈ -0.5*g
        a2, a1, a0 = np.polyfit(ts, zs, 2)
        # 取初速 vz0 = a1（理想情況 a2 ≈ -0.5*g；若偏差很大可混合 g 的先驗）
        vz0_quad = a1

        # z：方法B：先扣重力，再線性 fit，z'(t) = z + 0.5*g*t^2 ≈ vz0*t + z0
        z_prime = zs + 0.5*g*ts**2
        az, bz = np.polyfit(ts, z_prime, 1)
        vz0_lin = az

        # 兩者折衷（更穩定）
        vz0 = 0.5*vz0_quad + 0.5*vz0_lin

        return [ax, ay, vz0]
    
    def _predict_traj_until_pos(self, starting_point, optimized_V, fps, end_point,
                            alpha=0.242, base_dur=None, extra_time=0.2,
                            max_dur=2.0, dist_tol=0.50):
        """
        以『位置最接近 end_point(x,y,z)』為截斷標準。
        - 先模擬 base_dur + extra_time 的軌跡（不截地面）
        - 找到與 end_point 的最近點當截點
        - 若最近距離仍偏大，嘗試延長模擬（最多到 max_dur）
        回傳：N×4 (x,y,z,t)
        """
        if base_dur is None:
            base_dur = 1.0 / fps  # 至少一個步長，避免0秒

        def _simulate(dur):
            return physics_predict3d_v2(
                starting_point, optimized_V, fps,
                flight_time=dur, touch_ground_cut=False, alpha=alpha
            )

        # 1) 初次模擬：段長 + 緩衝
        dur = min(max(base_dur + extra_time, 1.0 / fps), max_dur)
        traj = _simulate(dur)
        if traj is None or len(traj) == 0:
            return traj

        def _cut_by_position(traj_np):
            xyz = traj_np[:, :3]
            end_xyz = np.array([end_point.x, end_point.y, end_point.z], dtype=float)
            dists = np.linalg.norm(xyz - end_xyz, axis=1)
            i_cut = int(np.argmin(dists))
            return i_cut, float(dists[i_cut])

        i_cut, best_dist = _cut_by_position(traj)

        # 2) 若距離仍大，逐步延長模擬時間再試（最多 max_dur）
        tried = {len(traj)}
        while (not isfinite(best_dist) or best_dist > dist_tol) and dur < max_dur:
            dur = min(dur + 0.2, max_dur)  # 每次多 0.2s
            traj_try = _simulate(dur)
            if traj_try is None or len(traj_try) in tried:
                break
            tried.add(len(traj_try))
            i_try, d_try = _cut_by_position(traj_try)
            if d_try < best_dist:
                traj, i_cut, best_dist = traj_try, i_try, d_try

        return traj[:i_cut + 1, :]

    def _traj_numpy_to_points(self, traj_np, base_fid, visibility=1):
        from lib.point import Point
        return [
            Point(fid=base_fid + i, visibility=visibility,
                x=float(p[0]), y=float(p[1]), z=float(p[2]),
                event=0, timestamp=float(p[3]))
            for i, p in enumerate(traj_np)
        ]
    
    def _traj2linear(self, traj_nd):
        # ndarray(N,4) → [{'id': i, 'pos': {'y': y, 'z': z, 'x': x}, 't': t}, ...]
        linear = []
        for i, (x, y, z, t) in enumerate(traj_nd):
            linear.append({'id': i, 'pos': {'x': float(x), 'y': float(y), 'z': float(z)}, 'timestamp': float(t)})
        return linear
    
    def publish_bridgegap(self, traj_fwd, traj_bwd, point_fwd, point_bwd, hit_pt):
        payload = {
            'traj_fwd': self._traj2linear(traj_fwd),
            'traj_bwd': self._traj2linear(traj_bwd),
            'point_fwd': self._traj2linear([point_fwd]),
            'point_bwd': self._traj2linear([point_bwd]),
            'hit_pt':   {
                'id': hit_pt.fid,
                'pos': {'x': hit_pt.x, 'y': hit_pt.y, 'z': hit_pt.z},
                'timestamp':  hit_pt.timestamp
            }
        }
        self._pub("BridgeGap/Debug", payload, "BridgeGap")

    def publish_servecheck(self, traj, is_serve, v=None, cfg=None):
        """
        將 is_valid_serve 的檢查結果（點列 + True/False）發佈出去，供 UI 顯示。
        Topic: /DATA/.../LayerContent/Model3D/ServeCheck/Debug
        """
        linear = []
        for p in traj:
            linear.append({
                'id': p.fid,
                'pos': {'x': float(p.x), 'y': float(p.y), 'z': float(p.z)},
                'timestamp': float(p.timestamp)
            })
        payload = {
            'type': 'serve_check',
            'is_serve': bool(is_serve),
            'start_fid': traj[0].fid if traj else None,
            'end_fid': traj[-1].fid if traj else None,
            'linear': linear,
            'v': [float(v[0]), float(v[1]), float(v[2])] if v is not None else None
        }
        self._pub("ServeCheck/Debug", payload, "ServeCheck")

    def publish_balltype(self, traj):
        linear = []
        for p in traj:
            linear.append({
                'id': p.fid,
                'pos': {'x': float(p.x), 'y': float(p.y), 'z': float(p.z)},
                'timestamp': float(p.timestamp)
            })
        payload = {
            'type': 'balltype_input',
            'linear': linear
        }
        self._pub("BallTypeInput/Debug", payload, "BallTypeInput")

class BaseState:
    name: DetectorState

    def on_enter(self, det: "FlyEventDetector", *args, **kwargs):
        # logging.info(f"Entering state: {self.name}")
        pass
    
    def on_exit(self, det: "FlyEventDetector", *args, **kwargs):
        # logging.info(f"Exiting state: {self.name}")
        pass

    def handler(self, det: "FlyEventDetector", traj, isFly, v, speed):
        logging.debug(f"[{self.name}] Handler called with traj={traj[0].fid} - {traj[-1].fid}, isFly={isFly}, v={v}, speed={speed}")
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def on_timeout(self, det):
        self._on_timeout(det)
        det.waiting_for_reset = False
        logging.debug(f"Reset waiting_for_reset")

    def _on_timeout(self, det):
        logging.debug(f"[TIMEOUT] {self.name} – no specific action")

class IdleState(BaseState):
    name = DetectorState.IDLE

    def on_enter(self, det, *args, **kwargs):
        super().on_enter(det, *args, **kwargs)

    def on_exit(self, det, *args, **kwargs):
        super().on_exit(det, *args, **kwargs)
    
    def handler(self, det, traj, isFly, v, speed):
        if v[2] > 0:
            det.transition_to_state(DetectorState.AWAIT_SERVE, traj=traj, v=v)
        else:
            logging.debug(f"[Idle State] No serve detected, waiting for serve")

    def _on_timeout(self, det):
        pass

class AwaitServeState(BaseState):
    name = DetectorState.AWAIT_SERVE

    def on_enter(self, det, *, traj, v, **kwargs):
        super().on_enter(det, traj=traj, v=v, **kwargs)
        # det.temp_serve_traj = list(traj)
        if det.temp_serve_traj:
            det.temp_serve_traj.extend(traj)
        else:
            det.temp_serve_traj = list(traj)
        det.temp_serve_v = v

        det.await_serve_prev_points.append(traj)
        det.await_serve_prev_v.append(v)
    
    def on_exit(self, det, *args, **kwargs):
        super().on_exit(det, *args, **kwargs)
        det.temp_serve_traj.clear()
        det.temp_serve_v = None
        det.await_serve_prev_points.clear()
        det.await_serve_prev_v.clear()
    
    def handler(self, det, traj, isFly, v, speed):
        direction_change = v[1] * det.temp_serve_v[1] < 0
        if direction_change:
            logging.debug(f"[Detect Serve] {det.temp_serve_traj[0].fid} - {det.temp_serve_traj[-1].fid}")
            if det.is_valid_serve(det.temp_serve_traj, det.temp_serve_v):
                logging.debug(f"[Valid Serve] {det.temp_serve_traj[0].fid} - {det.temp_serve_traj[-1].fid}")
                det.publisher.publish_event(2, det.temp_serve_traj[0])
                sendPoints(det.client, f"{det.topic}/Segment/Debug", det.temp_serve_traj)
                _ = det.publisher.publish_segment(det.temp_serve_traj)
                det.merged_segments.append((det.temp_serve_traj[0].fid, det.temp_serve_traj[-1].fid, det.temp_serve_v, list(det.temp_serve_traj)))
                logging.debug(f"[BridgeGap] Start {det.await_serve_prev_points[-1][-1].fid} - {traj[0].fid}")
                gap_hit_pt, gap_pts = det._bridge_gap(det.await_serve_prev_points[-1], det.await_serve_prev_v[-1], traj, v)
                if gap_hit_pt:
                    new_traj = [gap_hit_pt] + gap_pts + list(traj)      # ← 插入起始點
                    det.publisher.publish_event(1, gap_hit_pt)
                    det.update_prev_flight(new_traj, v)
                else:
                    det.publisher.publish_event(1, traj[0])
                    det.update_prev_flight(traj, v)
                det.transition_to_state(DetectorState.RALLY)
                # det.new_rally = False
                # det.update_prev_flight(traj, v)
                # det.event_triggered = False
                # det.creat_new_segment()  # 發球後的下一拍
            else: # Invalid serve, reset to 'IDLE' state
                det.transition_to_state(DetectorState.IDLE)
                # det.waiting_for_reset = True
                det.waiting_for_reset = False
                # det.temp_serve_traj = list(traj)
                # logging.debug(f"Init temp_serve_traj: {det.temp_serve_traj[0].fid} - {det.temp_serve_traj[-1].fid}")
                # det.temp_serve_v = v
                det.state.handler(det, traj, isFly, v, speed)
        else:
            det.temp_serve_traj.extend(traj)
            logging.debug(f"Extend temp_serve_traj: {det.temp_serve_traj[0].fid} - {det.temp_serve_traj[-1].fid}")
            det.await_serve_prev_points.append(traj)
            det.await_serve_prev_v.append(v)

    def _on_timeout(self, det):
        # await serve遇到timeout代表只有一拍，如果是發球得分會落地，如果z太高代表是傳球被接住
        serve = det.is_valid_serve(det.temp_serve_traj, det.temp_serve_v, publish=False)
        if serve:
            if det.temp_serve_traj[-1].z > 1.0:
                det.publisher.publish_servecheck(det.temp_serve_traj, False, det.temp_serve_v)
                det.reset_prev_flight()
                logging.debug(f'[Invalid Serve (timeout)] last point fid={det.temp_serve_traj[-1].fid}, z={det.temp_serve_traj[-1].z}')
                det.transition_to_state(DetectorState.IDLE)
                return
            det.publisher.publish_servecheck(det.temp_serve_traj, True, det.temp_serve_v)
            # serve -> segment -> dead
            logging.debug(f"[Valid Serve (timeout)] {det.temp_serve_traj[0].fid} - {det.temp_serve_traj[-1].fid}")
            # serve
            det.publisher.publish_event(2, det.temp_serve_traj[0])
            # segment
            sendPoints(det.client, f"{det.topic}/Segment/Debug", det.temp_serve_traj)
            pred_traj = det.publisher.publish_segment(det.temp_serve_traj, extend_to_ground=True)
            det.merged_segments.append((det.temp_serve_traj[0].fid, det.temp_serve_traj[-1].fid, det.temp_serve_v, list(det.temp_serve_traj)))
            # dead
            pred_dead_pt = Point(fid=det.temp_serve_traj[-1].fid, visibility=2, x=float(pred_traj[-1][0]), y=float(pred_traj[-1][1]), z=float(pred_traj[-1][2]), timestamp=float(float(pred_traj[-1][3])))
            det.publisher.publish_event(3, pred_dead_pt)
            logging.debug(f"[LandingMod] fid: {pred_dead_pt.fid+1}, t: {pred_dead_pt.timestamp}, pos: ({pred_dead_pt.x}, {pred_dead_pt.y}, {pred_dead_pt.z})")
            det.reset_prev_flight()
            det.transition_to_state(DetectorState.IDLE)
        else:
            det.publisher.publish_servecheck(det.temp_serve_traj, False, det.temp_serve_v)
            det.reset_prev_flight()
            det.transition_to_state(DetectorState.IDLE)


# TODO: 太短的Segment不要輸出
class RallyState(BaseState):
    name = DetectorState.RALLY

    def on_enter(self, det, *args, **kwargs):
        super().on_enter(det, *args, **kwargs)

    def on_exit(self, det, *args, **kwargs):
        super().on_exit(det, *args, **kwargs)

    def handler(self, det, traj, isFly, v, speed):
        new_traj = None
        if det.can_merge_segments(traj, v):
            logging.debug("... Merged with previous segment ... ") 
            det.merge_prev_flight(traj, v)
        else:
            # If not mergeable, finalize the previous segment first
            logging.debug(f"[BridgeGap] Start {det.prev_points[-1].fid} - {traj[0].fid}")
            # gap_hit_pt = det._bridge_gap(det.prev_points, det.prev_v, traj, v, max_gap=0.3)
            # gap_hit_pt, gap_pts = det._bridge_gap(det.prev_points, det.prev_v, traj, v)
            logging.debug(f"traj: {det.prev_traj[0].fid} - {det.prev_traj[-1].fid}, v: {det.prev_v}")
            gap_hit_pt, gap_pts = det._bridge_gap(det.prev_traj, det.prev_v, traj, v)
            # if len(det.prev_points) > 10:
            logging.debug(f"--> SHOT: NEW:{det.prev_start} - {det.prev_end}")
            sendPoints(det.client, f"{det.topic}/Segment/Debug", det.prev_points)
            _ = det.publisher.publish_segment(det.prev_points)
            det.merged_segments.append((det.prev_start, det.prev_end, det.prev_v, det.prev_points))
            if gap_hit_pt:
                new_traj = [gap_hit_pt] + gap_pts + list(traj)      # ← 插入起始點
                det.publisher.publish_event(1, gap_hit_pt)
                det.update_prev_flight(new_traj, v)
            else:
                det.publisher.publish_event(1, traj[0])
                det.update_prev_flight(traj, v)
            # else:
            #     logging.debug(f"Segment too short, not published: {det.prev_start} - {det.prev_end}")
        
        await_dead_traj = new_traj or traj
        if any(p.z < 1 for p in traj) and v[2] < 0:
            logging.debug(f"[Suspect Dead] Low point detected")
            lowest_point = min(traj, key=lambda p: p.z)
            logging.debug(f"Set lowest point: {lowest_point.fid}, {lowest_point.z}")
            logging.debug(f"[Rally State] Transition to AWAIT_DEAD: traj={await_dead_traj[0].fid} - {await_dead_traj[-1].fid}")
            det.transition_to_state(DetectorState.AWAIT_DEAD, traj=await_dead_traj, v=v)

    def _on_timeout(self, det):
        logging.debug(f"--> SHOT: DELAY: {det.prev_start} - {det.prev_end}")
        sendPoints(det.client, f"{det.topic}/Segment/Debug", det.prev_points)
        pred_traj = det.publisher.publish_segment(det.prev_points, extend_to_ground=True)
        det.merged_segments.append((det.prev_start, det.prev_end, det.prev_v, det.prev_points))
        # det.publisher.publish_event(3, det.prev_points[-1])
        pred_dead_pt = Point(fid=det.prev_points[-1].fid, visibility=2, x=float(pred_traj[-1][0]), y=float(pred_traj[-1][1]), z=float(pred_traj[-1][2]), timestamp=float(float(pred_traj[-1][3])))
        det.publisher.publish_event(3, pred_dead_pt)
        logging.debug(f"[LandingMod] fid: {pred_dead_pt.fid+1}, t: {pred_dead_pt.timestamp}, pos: ({pred_dead_pt.x}, {pred_dead_pt.y}, {pred_dead_pt.z})")
        det.reset_prev_flight()
        det.carryover_points.clear()
        det.transition_to_state(DetectorState.IDLE)


class AwaitDeadState(BaseState):
    name = DetectorState.AWAIT_DEAD

    def on_enter(self, det, *, traj, v, **kwargs):
        super().on_enter(det, traj=traj, v=v, **kwargs)
        det.temp_dead_traj = list(det.prev_points)
        # det.temp_dead_traj = list(traj)
        det.temp_dead_v = v
        det.await_dead_start_time = traj[0].timestamp
        det.await_dead_prev_points.append(traj)
        det.await_dead_prev_v.append(v)
    
    def on_exit(self, det, *args, **kwargs):
        super().on_exit(det, *args, **kwargs)
        det.temp_dead_traj.clear()
        det.temp_dead_v = None
        det.await_dead_prev_v.clear()
        det.await_dead_prev_points.clear()

    def handler(self, det, traj, isFly, v, speed):
        logging.debug(f"[Await Dead] {det.await_dead_prev_points[0][0].fid} - {det.await_dead_prev_points[-1][-1].fid}")
        # last_await_dead_points = det.await_dead_prev_points[-1]
        det.await_dead_prev_points.append(traj)
        det.await_dead_prev_v.append(v)
        for i, (traj, v) in enumerate(zip(det.await_dead_prev_points, det.await_dead_prev_v)):
            logging.debug(f"[Await Dead] await_dead_prev {i}: {traj[0].fid} - {traj[-1].fid}, v: {v}")
        # logging.debug(f"Extend await_dead_prev_points: {det.await_dead_prev_points[0][0].fid} - {det.await_dead_prev_points[-1][-1].fid}")
        
        upward = v[2] > 0
        # highest_point = max(traj, key=lambda p: p.z)
        # high_enough = highest_point.z > 0.5
        median_height = np.median([point.z for point in traj])
        high_enough = median_height > 0.5

        time_gap = traj[0].timestamp - det.temp_dead_traj[-1].timestamp
        internal_gap = any(
                (traj[i].timestamp - traj[i-1].timestamp) > 0.5 for i in range(1, len(traj))
            )
        if time_gap > 0.6 or internal_gap:
            logging.debug(f"[Await Dead] Time gap: ({traj[0].fid} - {det.temp_dead_traj[-1].fid}) -> {time_gap} -> Check dead point")
            logging.debug(f"temp_dead_traj: {det.temp_dead_traj[0].fid} - {det.temp_dead_traj[-1].fid}, temp_dead_v: {det.temp_dead_v}")
            dead_point, dead_idx = det.find_dead_point(det.temp_dead_traj)
            logging.debug(f"[Await Dead] Dead point: {dead_point.fid}, {dead_point.z}")

            dead_segment = det.temp_dead_traj[:dead_idx+1]
            sendPoints(det.client, f"{det.topic}/Segment/Debug", dead_segment)
            pred_traj = det.publisher.publish_segment(dead_segment, extend_to_ground=True)
            det.merged_segments.append((det.temp_dead_traj[0].fid, dead_segment[-1].fid, v, dead_segment))
            pred_dead_pt = Point(fid=dead_segment[-1].fid, visibility=2, x=float(pred_traj[-1][0]), y=float(pred_traj[-1][1]), z=float(pred_traj[-1][2]), timestamp=float(float(pred_traj[-1][3])))
            det.publisher.publish_event(3, pred_dead_pt)
            logging.debug(f"[LandingMod] fid: {pred_dead_pt.fid+1}, t: {pred_dead_pt.timestamp}, pos: ({pred_dead_pt.x}, {pred_dead_pt.y}, {pred_dead_pt.z})")
            det.reset_prev_flight()
            det.transition_to_state(DetectorState.IDLE)
            det.state.handler(det, traj, isFly, v, speed)
            logging.debug(f"[Await Dead] Handle {traj[0].fid} - {traj[-1].fid} at {det.state.name} after time gap")
        else:
            det.temp_dead_traj.extend(traj)
            det.temp_dead_v = v
            logging.debug(f"Extend temp_dead_traj: {det.temp_dead_traj[0].fid} - {det.temp_dead_traj[-1].fid}")

            if upward and high_enough:
                # logging.debug(
                #     (f"[Await Dead] Up-high trajectory detected, reset to RALLY state | "
                #      f"highest fid={highest_point.fid}, z={highest_point.z:.2f} m"
                #      f"upward speed={v[2]:.2f} m/s")
                # )
                logging.debug(
                    (f"[Await Dead] Up-high trajectory detected, reset to RALLY state | "
                     f"median height={median_height:.2f} m"
                     f"upward speed={v[2]:.2f} m/s")
                )
                for traj, v in zip(det.await_dead_prev_points[1:], det.await_dead_prev_v[1:]):
                    logging.debug(f"[Await Dead] reset RALLY {traj[0].fid} - {traj[-1].fid}, v: {v}")
                    if det.can_merge_segments(traj, v):
                        logging.debug("... Merged with previous segment ... ") 
                        det.merge_prev_flight(traj, v)
                    else:
                        # If not mergeable, finalize the previous segment first
                        logging.debug(f"--> SHOT: NEW:{det.prev_start} - {det.prev_end}")
                        sendPoints(det.client, f"{det.topic}/Segment/Debug", det.prev_points)
                        _ = det.publisher.publish_segment(det.prev_points)
                        det.merged_segments.append((det.prev_start, det.prev_end, det.prev_v, det.prev_points))
                        logging.debug(f"[BridgeGap] Start {det.prev_points[-1].fid} - {traj[0].fid}")
                        logging.debug(f"traj: {det.prev_traj[0].fid} - {det.prev_traj[-1].fid}, v: {det.prev_v}")
                        gap_hit_pt, gap_pts = det._bridge_gap(det.prev_traj, det.prev_v, traj, v)
                        if gap_hit_pt:
                            new_traj = [gap_hit_pt] + gap_pts + list(traj)      # ← 插入起始點
                            det.publisher.publish_event(1, gap_hit_pt)
                            det.update_prev_flight(new_traj, v)
                        else:
                            det.publisher.publish_event(1, traj[0])
                            det.update_prev_flight(traj, v)
                det.transition_to_state(DetectorState.RALLY)

    # TODO: 當前的temp_dead_traj裡可能有好幾個擊球，要做判斷，不能直接找死球
    def _on_timeout(self, det):
        dead_point, dead_idx = det.find_dead_point(det.temp_dead_traj)
        logging.debug(f"[Await Dead (delay)] Dead point: {dead_point.fid}, {dead_point.z}")
        dead_segment = det.temp_dead_traj[:dead_idx+1]
        sendPoints(det.client, f"{det.topic}/Segment/Debug", dead_segment)
        pred_traj = det.publisher.publish_segment(dead_segment, extend_to_ground=True)
        det.merged_segments.append((dead_segment[0].fid, dead_segment[-1].fid, det.await_dead_prev_v, dead_segment))
        # det.publisher.publish_event(3, dead_segment[-1])
        pred_dead_pt = Point(fid=dead_segment[-1].fid, visibility=2, x=float(pred_traj[-1][0]), y=float(pred_traj[-1][1]), z=float(pred_traj[-1][2]), timestamp=float(float(pred_traj[-1][3])))
        det.publisher.publish_event(3, pred_dead_pt)
        logging.debug(f"[LandingMod] fid: {pred_dead_pt.fid+1}, t: {pred_dead_pt.timestamp}, pos: ({pred_dead_pt.x}, {pred_dead_pt.y}, {pred_dead_pt.z})")
        det.reset_prev_flight()
        det.carryover_points.clear()
        det.transition_to_state(DetectorState.IDLE)


STATE_CLASS_MAP = {
    DetectorState.IDLE: IdleState(),
    DetectorState.AWAIT_SERVE: AwaitServeState(),
    DetectorState.RALLY: RallyState(),
    DetectorState.AWAIT_DEAD: AwaitDeadState(),
}


class flyEventDetector():
    def __init__(self, name, data_handler, client, topic, writer3D, fps, start_time, save_path='./'):
        self.name = name
        self.data_handler = data_handler
        self.client = client
        self.topic = topic
        self.writer3D = writer3D
        self.fps = fps
        self.start_time = start_time
        self.save_path = save_path

        # Publisher setup
        filePath_publish = f"{save_path}/Model3D_publish.csv"
        self.publisher = MQTTPublisher(self, self.data_handler, self.client, self.topic, CSVWriter(filePath_publish), self.fps)

        # Configuration
        cfg_override = {'fps': self.fps}
        base_cfg = DetectorConfig(fps=fps)
        self.cfg = replace(base_cfg, **(cfg_override or {}))



        self.points = [] # 3D point
        self.carryover_points = []
        self.lock = threading.Lock()
        self.last_point_time = time.time() # Time of the last received point
        self.last_ts = time.time()
        self.points_cleared = True  # Flag to indicate if points have been cleared
        
        self.alive = True
        self.delay_timer: threading.Timer | None = None
        # self._restart_timer()

        # segment and event management
        self.shots = []
        self.merged_segments = [] # Store merged segments
        self.prev_traj, self.prev_start, self.prev_end, self.prev_v, self.prev_points = None, None, None, None, []
        self.last_valid_time = None
        self.last_valid_fid = None
        self.event_triggered = False  # Flag to indicate if an event has been triggered
        self.new_rally = True

        # serve/dead temporary storage
        self.temp_serve_traj = []  # Temporary trajectory for serve
        self.temp_serve_v = None  # Temporary velocity for serve
        self.await_serve_prev_points, self.await_serve_prev_v = [], []  # Previous points and velocity for serve detection
        self.waiting_for_reset = False  # Flag to indicate if waiting for reset after serve
        self.temp_dead_traj = []  # Temporary trajectory for dead
        self.temp_dead_v = None # Temporary velocity for dead
        self.await_dead_prev_points, self.await_dead_prev_v = [], []  # Previous points and velocity for dead detection

        # State management
        self.state_enum = DetectorState.IDLE # Init state of the event detector
        self.state: BaseState = STATE_CLASS_MAP[self.state_enum]

        # Phase Label
        self.phase_records = []
        self.current_description = []
        self.current_phase = {
            "phase": "rest",
            "time_interval": [round(self.start_time, 3), None],
            "description": self.current_description
        }
        self.phase_records.append(self.current_phase)

        self.test_prev_v = None

    
    def transition_to_state(self, new_state: DetectorState, *args, **kwargs):
        """
        Transition to a new state and handle the exit of the current state and entry into the new state.
        """
        if new_state == self.state_enum:
            return
        logging.info(f"[State] : {self.state_enum.name} -> {new_state.name}")

        try:
            self.state.on_exit(self, *args, **kwargs)
        except Exception as e:
            logging.exception(f"on_exit error: {e}")

        self.state_enum = new_state
        self.state = STATE_CLASS_MAP[new_state]
        try:
            self.state.on_enter(self, *args, **kwargs)
        except Exception as e:
            logging.exception(f"on_enter error: {e}")

        self._restart_timer()

    def _start_new_phase(self, phase_type, start_time):
        self.current_description = []
        self.current_phase = {
            "phase": phase_type,
            "time_interval": [round(start_time, 3), None],
            "description": self.current_description
        }
        self.phase_records.append(self.current_phase)

    def _end_current_phase(self, end_time):
        if self.current_phase:
            self.current_phase["time_interval"][1] = round(end_time, 3)
            self.current_phase = None

    def _restart_timer(self):
        if self.delay_timer:
            self.delay_timer.cancel()
        delay_seconds = self.cfg.max_delay_by_state(self.state_enum)
        self.delay_timer = threading.Timer(delay_seconds, self._on_timeout)
        self.delay_timer.daemon = True
        self.delay_timer.start()
        self._delay_start_time = time.time()

    def _on_timeout(self):
        with self.lock:
            elapsed = time.time() - self._delay_start_time
            logging.debug(f"LONG DELAY ({self.state_enum.name}): {elapsed}")
            if len(self.points) > 0:
                logging.debug(f"Processing trajectory due to timeout: {self.points[0].fid} - {self.points[-1].fid}")
                self.process_trajectory(self.points)
                self.points = []
            logging.debug(f"Triggering on_timeout in current state - {self.state_enum.name}")
            self.state.on_timeout(self)
    
    def addPoint(self, point):
        """
        Add a new point to the detector.
        """
        self.points.append(point)
        self.last_point_time = time.time()
        self.last_ts = point.timestamp
        self._restart_timer()

    def detect(self):
        with self.lock:
            if len(self.points) >= self.cfg.sequence_length:
                self.points_cleared = False
                logging.info('')
                logging.info('========================================')
                logging.info(f"Sequence: {self.points[0].fid} - {self.points[-1].fid}")
                logging.debug('--------------------')
                self.process_trajectory(self.points)
                self.points = []
                self.points_cleared = True

    def process_trajectory(self, points):
        # logging.debug(f"--- Start process_trajectory --- {time.time() - self.start_time}")
        if self.carryover_points:
            heads_ts = points[0].timestamp if points else None
            tail_ts = self.carryover_points[-1].timestamp
            if heads_ts is not None and (heads_ts - tail_ts) <= self.cfg.carryover_max_gap:
                self.points = self.carryover_points + points
                logging.debug(f"Carryover points merged: {self.carryover_points[0].fid} - {self.carryover_points[-1].fid} with {self.points[0].fid} - {self.points[-1].fid}")
            else:
                logging.debug(f"Carryover points not merged: {self.carryover_points[0].fid} - {self.carryover_points[-1].fid} with {self.points[0].fid} - {self.points[-1].fid}")
            self.carryover_points = []

        results, new_carryover = trajectory_segment(
            self.points, fps=self.cfg.fps, initial_V=np.array([0, 0, 0]),
            loss_function=1, evaluation_method='vote',
            inlier_threshold=self.cfg.inlier_threshold, inlier_ratio=self.cfg.inlier_ratio,
            return_carryover=True)
        self.carryover_points = new_carryover

        for traj, isFly, v in results:
            speed = np.linalg.norm(v) if v is not None else -1
            self.trajectory_merged(traj, isFly, v, speed)
        # logging.debug(f"--- End process_trajectory --- {time.time() - self.start_time}")

    def trajectory_merged(self, traj, isFly, v, speed):
        logging.info('')
        if v is not None:
            logging.info(f"--> [{self.state_enum.name}] [{self.waiting_for_reset}] {traj[0].fid} - {traj[-1].fid} : {isFly}, Speed: {v}, {speed}")
        else:
            logging.info(f"--> [{self.state_enum.name}] [{self.waiting_for_reset}] {traj[0].fid} - {traj[-1].fid} : {isFly}")
        
        if self.waiting_for_reset:
            print("[Skip] Waiting for reset due to invalid serve")
            return
        
        traj_start_time = traj[0].timestamp
        if self.last_valid_time is not None and traj_start_time - self.last_valid_time > 3.0:
            logging.debug(f"... No trajectory detected for over 3 seconds, resetting ... ({self.last_valid_fid} - {traj[0].fid})")
            self.state.on_timeout(self)

        #     self.event_triggered = False
        #     self.new_rally = True
        #     self.transition_to_state(DetectorState.IDLE)
        #     self.waiting_for_reset = False  # 清除等待狀態

        if v is not None:
            if self.state_enum == DetectorState.AWAIT_DEAD:
                self.state.handler(self, traj, isFly, v, speed)
            else:
                median_height = np.median([point.z for point in traj])
                if isFly and(speed > self.cfg.flight_speed or (0 < speed < self.cfg.flight_speed and median_height > self.cfg.height_threshold)):
                    self.state.handler(self, traj, isFly, v, speed)
                else:
                    logging.debug(f"Don't handle trajectory 1: {traj[0].fid} - {traj[-1].fid}, isFly={isFly}")
        else:
            logging.debug(f"Don't handle trajectory 2: {traj[0].fid} - {traj[-1].fid}, v is None")

        self.last_valid_time = traj[-1].timestamp  # Update the timestamp of the last valid trajectory 
        self.last_valid_fid = traj[-1].fid  # Update the last valid frame ID
        self.test_prev_v = v

    def is_valid_serve(self, traj, v, publish=True):
        """
        Determine whether the given trajectory corresponds to a valid serve.

        Conditions:
            1. The starting point must exceed the serve-height threshold and lie within the allowed X/Y serve region.
            2. When the shuttle starts the negative-Y side, its initial motion must head toward +Y; conversly, a positive-Y start must head toward -Y.
            3. The initial vertical velocity v_z is positive (shuttle rising).
            4. The Z-coordinate should contain at most one strict peak.
            5. At least `min_points_for_serve` (default 10) samples are required.
            6. |ΔY| ≥ `min_serve_dy` (default 1.0 m)  ← NEW
        """
        if len(traj) < 2:  # Need at least two points to infer initial direction
            return False

        p = traj[0]
        
        # --- 1. Serve box & height check ----------------------------------------
        # in_z = p.z > self.cfg.serve_height
        h_min, h_max = self.cfg.serve_height
        in_z = h_min <= p.z <= h_max

        in_x = self.cfg.serve_region['x'][0] <= p.x <= self.cfg.serve_region['x'][1]
        in_y = (
            self.cfg.serve_region['y'][0] <= p.y <= self.cfg.serve_region['y'][1] or
            -self.cfg.serve_region['y'][1] <= p.y <= -self.cfg.serve_region['y'][0]
        )

        # --- 2. Y-direction toward opponent -------------------------------------
        towards_opponent = ((p.y < 0 and v[1] > 0) or (p.y > 0 and v[1] < 0))

        # --- 3. Vertical velocity upward ----------------------------------------
        logging.debug(f"Vertical velocity: {v[2]}")
        upward = v[2] > 0

        # --- 4. Exactly 0 or 1 peak in Z ----------------------------------------
        z_vals = np.array([p.z for p in traj])
        z_vals = gaussian_filter1d(z_vals, sigma=1)
        peak_ids, _ = find_peaks(z_vals, prominence=0.02, distance=5)
        logging.debug(f"Z peaks: {[traj[i].fid for i in peak_ids]}")
        single_peak = len(peak_ids) <= 1

        # --- 5. Minimum sample count -------------------------------------------
        points_num = len(traj) >= 10
        min_pts = getattr(self.cfg, "min_points_for_serve", 10)
        enough_points = len(traj) >= min_pts

        # --- 6. Y-displacement magnitude ----------------------------------------
        min_dy = getattr(self.cfg, "min_serve_dy", 1.0)       # metres
        dy_total = abs(traj[-1].y - p.y)
        big_enough_dy = dy_total >= min_dy

        is_serve = in_z and in_x and in_y and towards_opponent and upward and single_peak and enough_points and big_enough_dy
        logging.debug(f"[CHECK SERVE] {traj[0].fid} {traj[-1].fid} -> {is_serve} (x: {in_x}, y: {in_y}, z: {in_z}, num: {points_num}, up: {upward}, direction: {towards_opponent}, peak: {len(peak_ids)}, dy: {big_enough_dy})")
        if publish:
            self.publisher.publish_servecheck(traj, is_serve, v)
        return is_serve
    
    def find_dead_point(self, traj, eps=1e-3):
        logging.debug(f"Find dead point in trajectory: {traj[0].fid} - {traj[-1].fid}")
        n = len(traj)
        if n < 3:
            dead_point = traj[-1]
            return dead_point, n - 1
        
        z_vals = np.array([p.z for p in traj])
        z_vals = gaussian_filter1d(z_vals, sigma=1)

        dz = np.diff(z_vals)
        if np.all(dz >= -eps):
            dead_idx = n - 1
            logging.debug("Monotonically increasing (within eps); using last point as dead.")
            return traj[dead_idx], dead_idx

        apex_idx = int(np.argmax(z_vals))
        logging.debug(f"Apex index: {apex_idx}, fid: {traj[apex_idx].fid}, z: {z_vals[apex_idx]:.2f} m")

        tail = traj[apex_idx:]

        if len(tail) > 5:
            z_tail = np.array([p.z for p in tail])
            z_tail = gaussian_filter1d(z_tail, sigma=1)
            inv_z = -z_tail
            peaks, _ = find_peaks(inv_z, prominence=0.05, distance=2)
            logging.debug(f"Peaks: {[tail[i].fid for i in peaks]}")
            if len(peaks) > 0:
                logging.debug(f"Using first peak in tail: {tail[peaks[0]].fid}, z: {tail[peaks[0]].z:.2f} m")
                dead_idx = peaks[0] + apex_idx
            else:
                logging.debug(f"No peaks found in tail, using minimum Z value")
                dead_idx = apex_idx + int(np.argmin(z_tail))
        else:
            logging.debug(f"Tail too short, using minimum Z value in full trajectory")
            dead_idx = int(np.argmin(z_vals))
        logging.debug(f"Dead index: {dead_idx}, fid: {traj[dead_idx].fid}, z: {traj[dead_idx].z:.2f} m")
        dead_point = traj[dead_idx]
        return dead_point, dead_idx
    
    def can_merge_segments(self, traj, v, time_gap=2.0):
        # Same Y direction and time gap less than or equal to time_gap
        return (
            ((self.prev_v[1] >= 0 and v[1] >= 0) or (self.prev_v[1] < 0 and v[1] < 0)) and
            (traj[0].timestamp - self.prev_points[-1].timestamp <= time_gap)
        )
    
    def update_prev_flight(self, traj, v):
        self.prev_traj, self.prev_start, self.prev_end, self.prev_v, self.prev_points = traj, traj[0].fid, traj[-1].fid, v, list(traj)
        logging.debug(f"Update Prev {self.prev_points[0].fid} - {self.prev_points[-1].fid}")
    
    def reset_prev_flight(self):
        self.prev_traj, self.prev_start, self.prev_end, self.prev_v, self.prev_points = None, None, None, None, [] 

    def merge_prev_flight(self, traj, v):
        self.prev_traj = traj
        self.prev_end = traj[-1].fid
        self.prev_v = v
        self.prev_points.extend(traj)
        logging.debug(f"{self.prev_points[0].fid} - {self.prev_points[-1].fid}")

    def _shoot_forward(self, p, v, duration, fps, alpha=0.242):
        start = np.array([p.x, p.y, p.z, p.timestamp])
        return physics_predict3d_v2(start, v, fps, flight_time=duration, touch_ground_cut=False, alpha=alpha)

    def _bridge_gap(self, prev_traj, prev_v, next_traj, next_v, min_gap=5, max_gap=80, dist_th=5.0, alpha=0.242):
        """
        Bridge the gap between the last point of the previous trajectory and the first point of the next trajectory.
        """
        if not prev_traj or not next_traj:
            return None, None
        
        p_last = prev_traj[0]
        p_first = next_traj[0]
        gap_time = p_first.timestamp - prev_traj[-1].timestamp
        if gap_time <= min_gap/self.cfg.fps or gap_time > max_gap/self.cfg.fps:
            logging.debug(f"[BridgeGap] Gap too small({min_gap/self.cfg.fps}) or too large({max_gap/self.cfg.fps}): {gap_time:.3f}s")
            return None, None
        
        prev_traj_duration = prev_traj[-1].timestamp - prev_traj[0].timestamp
        traj_fwd = self._shoot_forward(p_last, prev_v, gap_time + prev_traj_duration, self.cfg.fps, alpha)
        traj_bwd = self._shoot_forward(p_first, -next_v, gap_time, self.cfg.fps, alpha)
        n = len(traj_bwd)
        times = p_first.timestamp - np.arange(n) / self.cfg.fps
        traj_bwd[:,3] = times
        traj_bwd = traj_bwd[::-1]  # Reverse the backward trajectory

        # 只取 prev_traj[-1] 之後的前向點
        tA_all, xyzA_all = traj_fwd[:, 3], traj_fwd[:, :3]
        valid_A = tA_all >= prev_traj[-1].timestamp
        if not np.any(valid_A):
            logging.debug("[BridgeGap] No forward samples after prev last timestamp.")
            return None, None
        tA  = tA_all[valid_A]
        xyzA = xyzA_all[valid_A]
        A_idx_all = np.flatnonzero(valid_A)  # 映回原始 traj_fwd 索引用

        tB, xyzB = traj_bwd[:,3], traj_bwd [:, :3]

        idxB = np.searchsorted(tB, tA)
        idxB = np.clip(idxB, 0, len(tB)-1)
        dists = np.linalg.norm(xyzA - xyzB[idxB], axis=1)
        k_local = np.argmin(dists)
        if dists[k_local] > dist_th:               # 超過門檻 → 補縫失敗
            logging.debug(f"[BridgeGap] Distance too large: {dists[k_local]:.3f}m")
            return None, None
        
        k_full = A_idx_all[k_local]         # ← 映回 traj_fwd 的「原始索引」
        t_hit  = tA[k_local]
        pA     = traj_fwd[k_full, :3]
        pB     = traj_bwd[idxB[k_local], :3]
        p_hit  = 0.5 * (pA + pB)            # 中點
        # logging.debug(
        #     "[BridgeGap] Hit candidate | "
        #     f"fwd_idx(local={k_local}, full={k_full}), t_fwd={t_hit:.6f}, pos_fwd={pA} | "
        #     f"bwd_idx={int(idxB[k_local])}, t_bwd={tB[idxB[k_local]]:.6f}, pos_bwd={pB} | "
        #     f"dist={dists[k_local]:.3f}m"
        # )
        new_fid = prev_traj[-1].fid + 1
        gap_hit_pt = Point(fid=new_fid, visibility=2, x=float(p_hit[0]), y=float(p_hit[1]), z=float(p_hit[2]), timestamp=t_hit)
        # logging.debug(f"[BridgeGap] Hit point: {gap_hit_pt.fid}, {gap_hit_pt.x}, {gap_hit_pt.y}, {gap_hit_pt.z}, {gap_hit_pt.timestamp}")
        
        # 產生 gap_pts：只取 (t_hit, p_first.timestamp) 的點
        mask_gap = (tB > t_hit) & (tB < p_first.timestamp)
        gap_rows = traj_bwd[mask_gap]

        gap_pts = []
        fid_gen = gap_hit_pt.fid + 1  # 避免和 hit_pt 重複
        for x, y, z, t in gap_rows:
            gap_pts.append(Point(fid=fid_gen, visibility=2,
                                x=float(x), y=float(y), z=float(z), timestamp=float(t)))
            fid_gen += 1

        self.publisher.publish_bridgegap(traj_fwd, traj_bwd, np.array(traj_fwd[k_full]), np.array(traj_bwd[idxB[k_local]]), gap_hit_pt)
        logging.debug(f"[BridgeGap] Bridged with {len(gap_pts)+1} points between fid {prev_traj[-1].fid} and {next_traj[0].fid}")
        return gap_hit_pt, gap_pts

        
        # # Calculate the time difference and distance
        # time_gap = first_point.timestamp - last_point.timestamp
        # dist = np.linalg.norm(np.array([first_point.x, first_point.y, first_point.z]) - np.array([last_point.x, last_point.y, last_point.z]))
        
        # if time_gap > max_gap or dist > dist_th:
        #     logging.debug(f"Bridging gap: {last_point.fid} -> {first_point.fid}, Time gap: {time_gap}, Distance: {dist}")
        #     # Predict the points to bridge the gap
        #     predicted_points = self._shoot_forward(last_point, v_next, time_gap, self.fps, alpha)
        #     return predicted_points + next_traj
        # else:
        #     return next_traj
    
    def save_json(self, json_path="label.json"):
        if self.current_phase and self.current_phase["time_interval"][1] is None:
            self._end_current_phase(self.last_ts)
            # self.current_phase["time_interval"][1] = round(time.time(), 3)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.phase_records, f, indent=4)
    
    def close(self):
        logging.debug('----- Event Detector Close -----')
        time.sleep(0.5)
        self.alive = False

        # if self.prev_v is not None and self.prev_points:
        #     # logging.debug(f"--> SHOT: {self.prev_start} - {self.prev_end}")
        #     self.state.on_timeout(self)

        # phase label
        start_time_str = datetime.fromtimestamp(self.start_time).strftime("%Y%m%d_%H%M")
        self.save_json(f"{self.save_path}/label_{start_time_str}.json")
        
        logging.debug("")
        logging.debug("===== SHOT =====")
        for idx, (s, e, v, traj) in enumerate(self.merged_segments, 1):
            logging.debug(f"{idx}: {traj[0].fid} - {traj[-1].fid}")


        