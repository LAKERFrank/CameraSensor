import paho.mqtt.client as mqtt
import logging
import ipaddress
import json
import signal
import uuid
import time
import matplotlib.pyplot as plt
import queue
import argparse
import random
import numpy as np
import threading
import colorsys
import matplotlib.text as mtext
import matplotlib.colors as mcolors
import glob, os

from datetime import datetime
from matplotlib.widgets import Button, RadioButtons, TextBox

from lib.common import ROOTDIR


class ExampleAPP():

    def __init__(self, broker_ip: str = None, broker_port: str = None, max_points: int = None, office: bool = False):
        self.app_uuid = str(uuid.uuid4())
        self.target_devices_name = "ContentDevice"
        self.has_explored = False
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_message = self.on_message

        self.CONTROL_PLANE_QOS = 2

        if broker_ip is None:
            self.mqttc.connect("140.113.208.125", 1884, 60)
        else:
            try:
                ipaddress.ip_address(broker_ip)
                self.mqttc.connect(broker_ip, broker_port)
            except:
                logging.error("Fail to connect MQTT Broker.")

        self.mqttc.loop_start()

        self.SUBS = [
            "/DATA/{}/ContentLayer/Model3D/Point",
            "/DATA/{}/ContentLayer/Model3D/Event/Debug",
            "/DATA/{}/ContentLayer/Model3D/Segment/Debug",
            "/DATA/{}/ContentLayer/Model3D/BridgeGap/Debug",
            "/DATA/{}/ContentLayer/Model3D/ServeCheck/Debug",
            "/DATA/{}/ContentLayer/Model3D/Landing/Debug",
            "/DATA/{}/ContentLayer/Model3D/BallTypeInput/Debug",
            "/DATA/{}/ContentLayer/Model3D/Event",
            "/DATA/{}/ContentLayer/Model3D/Segment",
        ]

        self.new_data_received = False

        # Initialize plot with two subplots: one for plotting and one for text display on the right side
        self.fig, (self.ax, self.ax_text) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4.5, 1.5]}, figsize=(12, 6))
        self.ax_overlay = self.fig.add_axes(self.ax.get_position(), facecolor='none')
        self.ax_overlay.set_axis_off()
        self.ax_overlay.set_zorder(1000)  # ★ 最上層
        self.office = office
        self.set_axis()

        self.fig.canvas.mpl_connect('button_press_event', self._on_click_fallback)
        self.fig.canvas.mpl_connect('resize_event', self._sync_overlay_position)

        # rally management
        self.current_rally_id: int = 0      # updated on Serve
        self.selected_rally_id: int = 0     # 0 → All
        self.segment_data: list[dict] = []  # {segment_id, points, color, rally_id}
        self.segment_counter = 0

        self.text_lines = []  # Store text lines for segment information
        self.line_idx_to_segment: dict[int, int] = {}  # Map line index to segment ID

        self.selected_servecheck_idx = None
        self.highlight_sc_artists = []   # ServeCheck 的 highlight 圈圈
        # 你已經有 pick_event 綁 on_pick_segment，我們會擴充它處理 ServeCheck
        if not hasattr(self, 'line_idx_to_servecheck'):
            self.line_idx_to_servecheck = {}

        # Scatter plots for different topics:
        # "point" for normal points (使用透明灰色)
        self.scat_point = self.ax.scatter([], [], s=5, c='grey', alpha=0.3, label='Model3D_point')
        # "event" for event points (使用紅色)
        self.scat_event = self.ax.scatter([], [], s=40, c='red', label='Model3D_event')
        self.event_labels = []  # 用於存放 event 序號
        self.scat_event_hit = self.ax.scatter([], [], s=40, c='blue', label='Event 1 (Hit)')
        self.scat_event_serve = self.ax.scatter([], [], s=40, c='red', label='Event 2 (Serve)')
        self.scat_event_dead = self.ax.scatter([], [], s=40, c='green', label='Event 3 (Dead)')
        self.segment_artists = []  # keep references so we can remove
        self.event_texts = []
        self._bridge_artists = []
        self.bridge_gaps = []
        # Serve 檢查結果（True/False）各自一層
        self.serve_checks = []  # list of dict: {'ok': bool, 'points': [(y,z), ...], 'start_fid':..., 'end_fid':..., 'rally_id': int}
        self.scat_serve_ok = self.ax.scatter([], [], s=12, marker='^', c='tab:red',  label='ServeCheck OK',  alpha=0.9, zorder=4)
        self.scat_serve_ng = self.ax.scatter([], [], s=12, marker='x', c='tab:gray', label='ServeCheck NG',  alpha=0.9, zorder=4)

        self._landing_artists = []   # 存放預測軌跡與落地點的可視化物件
        self.landings = []           # 可序列化資料（存進 session 用）

        # BallTypeInput：給 detectBallTypeDT 的輸入點（預測/擴展後的 points_for_balltype）
        self.scat_balltype = self.ax.scatter([], [], s=5, marker='v', c='orange',
                                            label='BallTypeInput', alpha=0.3, zorder=5)
        self.balltype_inputs = []   # list of dicts: {'points': [(y,z), ...], 'meta': [...], 'rally_id': int}

        # --- Hover tooltip for points ---
        self.hover_annot = self.ax_overlay.annotate(
            "", xy=(0,0), xytext=(12, 12),
            xycoords=self.ax.transData,     # ★ 仍然用主圖的資料座標當錨點
            textcoords="offset points",
            ha='left', va='bottom',
            bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.95),
            arrowprops=dict(arrowstyle="->", lw=0.5, alpha=0.8),
            zorder=999,
            annotation_clip=False,
        )
        self.hover_annot.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # 讓 scatter 容易被滑鼠命中（提高 pick 半徑）
        self.scat_point.set_picker(5)
        self.scat_event.set_picker(5)
        self.scat_event_hit.set_picker(5)
        self.scat_event_serve.set_picker(5)
        self.scat_event_dead.set_picker(5)

        self.selected_segment_id = None      # 目前被選中的 segment_id
        self.highlight_artist = None         # 用來繪製高亮覆蓋層
        self.fig.canvas.mpl_connect('pick_event', self.on_pick_segment)

        # Buffers for different topics
        self.max_points = max_points
        self.point_buffers = {
            'point': queue.Queue(maxsize=max_points) if max_points else [],
            'event': queue.Queue(maxsize=max_points) if max_points else [],
        }
        self.event_buffers = {
            1: queue.Queue(maxsize=max_points or 0),
            2: queue.Queue(maxsize=max_points or 0),
            3: queue.Queue(maxsize=max_points or 0),
        }

        # For trajectory segments (Segment)
        self.segment_colors = []  # Store colors for segments
        self.segment_points = []  # Store points for segments
        self.segment_counter = 0
        self.segment_id = 0
        self.segment_data = []  # Save segment data (id, points, fid)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        # ↓ Latest button
        ax_go_bottom = self.fig.add_axes([0.9, 0.05, 0.05, 0.03])
        self.btn_go_bottom = Button(ax_go_bottom, '↓ Latest')
        self.btn_go_bottom.on_clicked(self.on_go_bottom_clicked)

        self.text_offset = 0
        self.text_display_count = 35
        self.line_step = 0.034

        self.follow_latest = True

        # Clear button
        self.ax_button = self.fig.add_axes([0.9, 0.95, 0.05, 0.04])
        self.button = Button(self.ax_button, 'Clear')
        self.button.on_clicked(self.clear_data)

        # Button to toggle bridgegap visibility
        self.show_bridgegap = True
        ax_bg = self.fig.add_axes([0.805, 0.95, 0.09, 0.04])
        self.btn_bg = Button(ax_bg, 'BridgeGap: ON')
        self.btn_bg.on_clicked(self._toggle_bridgegap)

        # Button to toggle BalltypeInput visibility
        self.show_balltype = True
        ax_bt = self.fig.add_axes([0.715, 0.95, 0.085, 0.04])  # 位置可自行微調
        self.btn_balltype = Button(ax_bt, 'BallType: ON')
        self.btn_balltype.on_clicked(self._toggle_balltype)

        # RadioButtons for rally selection – start with only "All"
        self.ax_rally = self.fig.add_axes([0.66, 0.1, 0.05, 0.8])
        self.rally_radio = RadioButtons(self.ax_rally, ['All'], active=0)
        self.rally_radio.on_clicked(self.on_rally_selected)

        # Last received time
        self.ax_last_received = self.fig.add_axes([0.65, 0.9, 0.1, 0.05])
        self.ax_last_received.axis('off')
        self.last_received_text = self.ax_last_received.text(0.5, 0.5, "Last received: --:--:--",
                                                             fontsize=10, color='orange', ha='center', va='center')

        self.last_received_time = time.time() # Store the time of the last received point message
        self.no_message_threshold = 10 # seconds
        # Start the timer thread
        self.timer_thread = threading.Thread(target=self.check_for_no_point_messages)
        self.timer_thread.daemon = True
        self.timer_thread.start()

        self.EVENT_DESCRIPTIONS = {
            1: "Hit",           # 擊球
            2: "Serve",         # 發球
            3: "Dead",          # 死球
            4: "Non-FreeFly",   # 不自由飛行
            5: "NoBall/Static"  # 無球/靜止
        }

        # —— Save 按鈕 ——（放在輸入框右邊）
        ax_save = self.fig.add_axes([0.36, 0.95, 0.08, 0.04])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self._on_save_session)

        # —— 檔名輸入框（常駐在視窗上方，初始為空）——
        ax_namebox = self.fig.add_axes([0.20, 0.95, 0.15, 0.04])  # [left, bottom, width, height]
        self.save_name_tb = TextBox(ax_namebox, label='Save Filename: ', initial="")
        try:
            self.save_name_tb.ax.set_facecolor((1, 1, 1, 0.9))  # 讓底色更明顯（可選）
        except Exception:
            pass

        # ====== 檢視（View）輸入框 + Load 按鈕 ======
        ax_viewbox = self.fig.add_axes([0.20, 0.90, 0.15, 0.04])   # [left, bottom, width, height]
        self.view_name_tb = TextBox(ax_viewbox, label='View: ', initial="")
        try:
            self.view_name_tb.ax.set_facecolor((1, 1, 1, 0.9))
        except Exception:
            pass

        ax_load = self.fig.add_axes([0.36, 0.90, 0.08, 0.04])
        self.btn_load = Button(ax_load, 'Load')
        self.btn_load.on_clicked(self._on_load_session_from_ui)

        # ====== 模式切換（Live / View）======
        self.mode = 'live'               # 'live' 或 'view'
        self.ignore_mqtt = False         # True 時忽略 on_message 資料

        ax_mode = self.fig.add_axes([0.46, 0.93, 0.05, 0.06])  # 小小一塊給 RadioButtons
        self.mode_radio = RadioButtons(ax_mode, ['Live', 'View'], active=0)
        self.mode_radio.on_clicked(self._on_mode_changed)

        # 顯示 Mode 狀態
        self.ax_mode_label = self.fig.add_axes([0.52, 0.94, 0.2, 0.04])
        self.ax_mode_label.axis('off')
        self.mode_text = self.ax_mode_label.text(0, 0.5, "Mode: Live", fontsize=10, color='tab:blue', ha='left', va='center')
        self.current_loaded_session = None

        # ====== FID 搜尋 ======
        ax_fid = self.fig.add_axes([0.49, 0.89, 0.16, 0.03])  # [left, bottom, width, height]
        self.fid_tb = TextBox(ax_fid, label='Find fid: ', initial="")
        try:
            self.fid_tb.ax.set_facecolor((1, 1, 1, 0.9))
        except Exception:
            pass
        self.fid_tb.on_submit(self._on_fid_submit)

        self._fid_query = None              # 解析後的查詢條件
        self._fid_highlight_art = None      # 高亮覆蓋層（Scatter）
        self._fid_highlight_texts = []      # 對應的 fid 文字標籤

        self.ax.set_zorder(30)
        self.ax_text.set_zorder(20)
        for _ax in [self.ax_rally, self.ax_button, self.ax_last_received,
                    ax_bg, ax_bt, ax_go_bottom, ax_save, ax_namebox, ax_viewbox,
                    ax_load, ax_mode, self.ax_mode_label, ax_fid]:
            _ax.set_zorder(5)



    def on_connect(self, client: mqtt.Client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code}")

    def on_message(self, client, userdata, msg):
        self.new_data_received = True
        # Process based on topic
        if msg.topic == f"/DATA/{self.target_devices_name}/ContentLayer/Model3D/Point":
            self.process_point_message(msg)
            # print("=== point ===")
        elif msg.topic == f"/DATA/{self.target_devices_name}/ContentLayer/Model3D/Event/Debug":
            print(f"[ApplicationLayer] Received on topic '{msg.topic}'")
            self.process_event_message_debug(msg)
        elif msg.topic == f"/DATA/{self.target_devices_name}/ContentLayer/Model3D/Segment/Debug":
            print(f"[ApplicationLayer] Received on topic '{msg.topic}'")
            self.process_segment_message_debug(msg)
        
        elif msg.topic == f"/DATA/{self.target_devices_name}/ContentLayer/Model3D/Event":
            print()
            print(f"Event")
            print(datetime.now().strftime("%H:%M:%S"))
            print(msg.payload)
            self.process_event_message(msg)

        elif msg.topic == f"/DATA/{self.target_devices_name}/ContentLayer/Model3D/Segment":
            print()
            print(f"Segment")
            print(datetime.now().strftime("%H:%M:%S"))
            print(msg.payload)
            self.process_segment_message(msg)

        elif msg.topic == f"/DATA/{self.target_devices_name}/ContentLayer/Model3D/BridgeGap/Debug":
            print(f"[ApplicationLayer] Received on topic '{msg.topic}'")
            # print(msg.payload)
            self.process_bridgegap_message(msg)

        elif msg.topic == f"/DATA/{self.target_devices_name}/ContentLayer/Model3D/ServeCheck/Debug":
            print(f"[ApplicationLayer] Received on topic '{msg.topic}'")
            # print(msg.payload)
            self.process_servecheck_message(msg)

        elif msg.topic == f"/DATA/{self.target_devices_name}/ContentLayer/Model3D/Landing/Debug":
            print(f"[ApplicationLayer] Received on topic '{msg.topic}'")
            self.process_landing_message(msg)

        elif msg.topic == f"/DATA/{self.target_devices_name}/ContentLayer/Model3D/BallTypeInput/Debug":
            self.process_balltype_input_message(msg)

        else:
            print('---')
            print(msg.topic)
            print(msg.payload)

    
    def set_axis(self):
        self.ax.set_xlabel('Y Coordinate')
        self.ax.set_ylabel('Z Coordinate')
        
        if self.office:
            self.x_min = -3
            self.x_max = 3
            self.y_min = 0
            self.y_max = 5
        else:
            self.x_min = -7
            self.x_max = 7
            self.y_min = 0
            self.y_max = 6

        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_xticks(range(self.x_min, self.x_max+1))
        self.ax.set_yticks(range(self.y_min, self.y_max+1))
        self.ax_text.axis('off')

    
    def check_for_no_point_messages(self):
        """Check every second if no message has been received for 3 minutes."""
        while True:
            time.sleep(10)  # Check every second
            if time.time() - self.last_received_time > self.no_message_threshold:
                # self.update_text_display_no_point_message()
                self._need_update_clock = True
            # else:
            #     self.update_text_display()  # Update with regular info
                
    
    def update_text_display_no_point_message(self):
        """Display the current time when no point message is received for 3 minutes."""
        if self.mode == 'view':
            return
        # self.ax_text.cla()  # Clear previous text
        # self.ax_text.axis('off')
        current_time = datetime.now().strftime("%H:%M:%S")
        last_received_formatted = datetime.fromtimestamp(self.last_received_time).strftime("%H:%M:%S")
        
        # self.ax_text.text(0.01, 1 - 0.03, f"Last received time: {current_time}", fontsize=10, color='orange', ha='left', va='top')
        # self.fig.canvas.draw_idle()

        self.last_received_text.set_text(f"Current Time: {current_time}, Last received: {last_received_formatted}")
        # self.fig.canvas.draw_idle()
        # self.ax_last_received.figure.canvas.draw_idle()

    def process_point_message(self, msg):
        """Process individual point messages for Model3D points."""
        self.last_received_time = time.time()
        # print(self.last_receiver_time)
        # print(time.time())
        # print(datetime.fromtimestamp(self.last_received_time).strftime("%H:%M:%S"))

        payload = json.loads(msg.payload)
        for point in payload.get('linear', []):
            x = point['pos']['x']
            y = point['pos']['y']
            z = point['pos']['z']
            ts = point.get('timestamp', None)
            fid = point.get('id', 'NoID')

            # Add new point to the buffer
            item = (x, y, z, fid, ts)
            if self.max_points:
                if self.point_buffers['point'].full():
                    self.point_buffers['point'].get()  # Remove the oldest point
                self.point_buffers['point'].put(item)
            else:
                self.point_buffers['point'].append(item)  # Keep all points when no limit

    def process_event_message_debug(self, msg):
        """Process debug event messages with colored points and labels."""
        payload = json.loads(msg.payload)
        for point in payload.get('linear', []):
            y = point['pos']['y']
            z = point['pos']['z']
            fid = point.get('id', 'NoID')

            # Add new event point to the buffer
            if self.max_points:
                if self.point_buffers['event'].full():
                    self.point_buffers['event'].get()  # Remove the oldest point
                self.point_buffers['event'].put((y, z))
            else:
                self.point_buffers['event'].append((y, z))  # Keep all points when no limit

            # 記錄序號與 ID
            seq = len(self.event_labels) + 1  # Event 序號
            self.event_labels.append((seq, fid))
            print(self.event_labels, fid)        


    def process_segment_message_debug(self, msg):
        print('Segment Debug')
        """Process debug segment messages for Segment."""
        payload = json.loads(msg.payload)
        trajectory = payload.get('linear', [])

        print(f"{trajectory[0]['id']} - {trajectory[-1]['id']}")

        # Increment segment counter and assign a unique ID
        self.segment_counter += 1
        self.segment_id = self.segment_counter

        # Generate a random color for the new segment
        segment_color = (random.random(), random.random(), random.random())

        # Extract points and fid from the segment
        segment_points = []
        segment_fids = []
        segment_meta = []
        for point in trajectory:
            x = point['pos']['x']
            y = point['pos']['y']
            z = point['pos']['z']
            ts = point.get('timestamp', None)
            fid = point.get('id', 'N/A')
            segment_points.append((y, z))
            segment_fids.append(fid)
            segment_meta.append({'x': x, 'y': y, 'z': z, 'fid': fid, 'ts': ts})

        # Save segment data for logging or exporting
        self.segment_data.append({
            "segment_id": self.segment_id,
            "points": segment_points,
            "fids": segment_fids,
            "meta": segment_meta,
            "color": segment_color,
            "rally_id": self.current_rally_id or 0
        })

        # For visualization
        self.segment_colors.append(segment_color)
        self.segment_points.append(segment_points)

        # Update right-side text for segments (顯示每個 Segment 的第一個和最後一個點的 fid)
        self.update_text_display()

    
    def process_event_message(self, msg):
        """Process Event Publish messages and update text display."""
        payload = json.loads(msg.payload)
        fid = payload.get("fid", "N/A")
        event_type = payload.get("event", None)
        position = payload.get("position")
        ts = payload.get("timestamp", None)

        if event_type == 2:
            self.current_rally_id += 1
            self.update_rally_buttons()
            new_line = f"--------------- Rally {self.current_rally_id} ---------------"
            self.text_lines.append(new_line)
        rally_id = self.current_rally_id if self.current_rally_id else 0

        if event_type in self.event_buffers and position:
            x, y, z = position[0], position[1], position[2]
            data = (x, y, z, fid, rally_id, ts)
            buffer = self.event_buffers[event_type]
            if self.max_points:
                if buffer.full():
                    buffer.get()
                buffer.put(data)
            else:
                buffer.put(data)

        event_desc = self.EVENT_DESCRIPTIONS.get(event_type, f"Unknown({event_type})")
        new_line = f">> Event: {fid}, Type: {event_desc}"
        self.text_lines.append(new_line)

        if event_type == 3:
            new_line = f"------------------------------------------"
            self.text_lines.append(new_line)

        self.update_text_display()
        
    
    def process_segment_message(self, msg):
        """Process Segment Publish messages to calculate combined speed and update text display."""
        payload = json.loads(msg.payload)
        start_fid = payload.get("start_fid", "N/A")
        end_fid = payload.get("end_fid", "N/A")
        speed = payload.get("speed", [0, 0, 0])
        ball_type = payload.get("ball_type", "N/A")

        # Calculate combined speed
        combined_speed = (speed[0]**2 + speed[1]**2 + speed[2]**2) ** 0.5 # m/s
        combined_speed *= 3.6 # km/hr

        new_line = f"(R{self.current_rally_id or '-'}) Segment : {start_fid} - {end_fid}, Speed: {combined_speed:.2f}, Ball Type: {ball_type}"

        # Append the new line to text_lines and update text display
        self.text_lines.append(new_line)

        self.line_idx_to_segment[len(self.text_lines) - 1] = self.segment_id

        self.update_text_display()

    def process_bridgegap_message(self, msg):
        """ Draw fwd / bwd / hit_pt """
        payload = json.loads(msg.payload)

        # ① 讀取 y、z、ts
        def lin2yzts(lin):
            y, z, ts = [], [], []
            for p in lin:
                y.append(p['pos']['y'])
                z.append(p['pos']['z'])
                ts.append(p.get('timestamp'))          # ← 依你 JSON 的欄位名稱調整
            return np.array(y), np.array(z), ts

        y_fwd, z_fwd, ts_fwd = lin2yzts(payload['traj_fwd'])
        y_bwd, z_bwd, ts_bwd = lin2yzts(payload['traj_bwd'])

        p_fwd = payload['point_fwd'][0]
        py_fwd = p_fwd['pos']['y']
        pz_fwd = p_fwd['pos']['z']
        pt_fwd = p_fwd.get('timestamp')

        p_bwd = payload['point_bwd'][0]
        py_bwd = p_bwd['pos']['y']
        pz_bwd = p_bwd['pos']['z']
        pt_bwd = p_bwd.get('timestamp')

        hit         = payload['hit_pt']
        y_hit       = hit['pos']['y']
        z_hit       = hit['pos']['z']
        ts_hit      = hit.get('timestamp')

        bridge_objs = []
        # ② 畫 scatter 點
        # art_fwd = self.ax.scatter(y_fwd, z_fwd, s=5, c='cyan',    marker='o', label='traj_fwd')
        # art_bwd = self.ax.scatter(y_bwd, z_bwd, s=5, c='magenta', marker='o', label='traj_bwd')
        art_fwd, = self.ax.plot(y_fwd, z_fwd, '-o', color='cyan', markersize=3, linewidth=0.8)
        art_bwd, = self.ax.plot(y_bwd, z_bwd, '-o', color='magenta', markersize=3, linewidth=0.8)
        art_p_fwd = self.ax.scatter([py_fwd], [pz_fwd], s=40, c='orange', marker='o',
                                    zorder=3, label='point_fwd')
        art_p_bwd = self.ax.scatter([py_bwd], [pz_bwd], s=40, c='orange', marker='o',
                                    zorder=3, label='point_bwd')
        art_hit = self.ax.scatter([y_hit], [z_hit], s=80, c='orange', marker='*',
                                zorder=4, label='hit_pt')
        bridge_objs.append(art_fwd)
        bridge_objs.append(art_bwd)
        bridge_objs.append(art_p_fwd)
        bridge_objs.append(art_p_bwd)
        bridge_objs.append(art_hit)
        for art in bridge_objs:
            art.set_zorder(15)

        # ③ 為每點加上 t 標籤
        def annotate_pts(y_arr, z_arr, ts_arr, color):
            for y, z, ts in zip(y_arr, z_arr, ts_arr):
                if ts is None:
                    continue
                tstr = f"{ts:.3f}"                  # 直接顯示數值；要 HH:MM:SS 可自行轉換
                txt = self.ax.text(y, z - 0.15, tstr,
                                fontsize=6, color=color, ha='center', va='top')
                self._bridge_artists.append(txt)    # ← 讓 Clear 時一起刪
                bridge_objs.append(txt)

        # annotate_pts(y_fwd, z_fwd, ts_fwd, color='cyan')
        # annotate_pts(y_bwd, z_bwd, ts_bwd, color='magenta')

        if ts_hit is not None:
            tstr = f"{ts_hit:.3f}"
            txt_hit = self.ax.text(y_hit, z_hit - 0.2, tstr,
                                fontsize=9, color='orange', ha='center', va='top')
            self._bridge_artists.append(txt_hit)
            bridge_objs.append(txt_hit)

        # ④ 把 scatter 物件也收進 _bridge_artists（供 Clear）
        # self._bridge_artists.extend([art_fwd, art_bwd, art_p_fwd, art_p_bwd, art_hit])
        r_id = self.current_rally_id or 0
        for art in bridge_objs:
            art.rally_id = r_id
        self._bridge_artists.extend(bridge_objs)

        self._refresh_bridgegap_visibility()       # ★ 新增
        self.fig.canvas.draw_idle()                # ★ 新增
        
        # ★ 存一份可序列化的快照（只存必要欄位）
        self.bridge_gaps.append({
            'rally_id': r_id,
            'fwd':  [{'y': float(y), 'z': float(z), 't': (ts_fwd[i] if i < len(ts_fwd) else None)}
                    for i, (y, z) in enumerate(zip(y_fwd, z_fwd))],
            'bwd':  [{'y': float(y), 'z': float(z), 't': (ts_bwd[i] if i < len(ts_bwd) else None)}
                    for i, (y, z) in enumerate(zip(y_bwd, z_bwd))],
            'p_fwd': {'y': float(py_fwd) if py_fwd is not None else None,
                    'z': float(pz_fwd) if pz_fwd is not None else None,
                    't': pt_fwd},
            'p_bwd': {'y': float(py_bwd) if py_bwd is not None else None,
                    'z': float(pz_bwd) if pz_bwd is not None else None,
                    't': pt_bwd},
            'hit':   {'y': float(y_hit) if y_hit is not None else None,
                    'z': float(z_hit) if z_hit is not None else None,
                    't': ts_hit}
        })
        
        # ⑤ 通知主迴圈刷新
        self.new_data_received = True
    
    def process_servecheck_message(self, msg):
        """顯示 is_valid_serve 檢查結果。True: 三角形（不畫），False: 叉叉（會畫）。右側文字列出全部。"""
        payload = json.loads(msg.payload)
        ok = bool(payload.get('is_serve', False))
        linear = payload.get('linear', [])
        start_fid = payload.get('start_fid', None)
        end_fid = payload.get('end_fid', None)

        # 取 y, z + meta（fid/ts）
        pts = []
        meta = []
        for p in linear:
            pos = p.get('pos', {})
            y = pos.get('y', None)
            z = pos.get('z', None)
            if y is None or z is None:
                continue
            pts.append((y, z))
            meta.append({
                'y': y,
                'z': z,
                'fid': p.get('id', None),
                'ts': p.get('timestamp', None),
            })

        r_id = self.current_rally_id or 0
        self.serve_checks.append({
            'ok': ok,
            'points': pts,
            'meta': meta,
            'start_fid': start_fid,
            'end_fid': end_fid,
            'rally_id': r_id
        })

        # 在右側面板新增一行（只是加到資料，繪製交給 update_text_display）
        verdict = "OK" if ok else "NG"
        new_line = f"[ServeCheck {verdict}] {start_fid} - {end_fid}"
        self.text_lines.append(new_line)

        # 把此行綁定到 servecheck 索引，稍後 update_text_display 會把它設成可點擊
        if not hasattr(self, 'line_idx_to_servecheck'):
            self.line_idx_to_servecheck = {}
        self.line_idx_to_servecheck[len(self.text_lines) - 1] = len(self.serve_checks) - 1

        self.update_text_display()
        self.new_data_received = True  # 觸發 update_plot()

    def process_landing_message(self, msg):
        payload = json.loads(msg.payload)
        pred = payload.get('pred_linear', [])
        if not pred:
            return

        yy = [p['pos']['y'] for p in pred]
        zz = [p['pos']['z'] for p in pred]
        ts = [p.get('timestamp') for p in pred]

        # 讀落地點
        landing = payload.get('landing', {})
        lpos = landing.get('pos', {})
        ly, lz = lpos.get('y'), lpos.get('z')
        lt = landing.get('timestamp', None)
        end_fid = payload.get('end_fid', None)

        # 畫預測軌跡（虛線 + 小點）
        line_pred, = self.ax.plot(yy, zz, '--', linewidth=1.0, alpha=0.5, zorder=3)
        scat_pred = self.ax.scatter(yy, zz, s=3, alpha=0.5, zorder=4)

        # 標示落地點（用星形比較醒目）
        scat_landing = self.ax.scatter([ly], [lz], s=90, marker='*',
                                        facecolors='none', edgecolors='lime', linewidths=1.8,
                                        zorder=5)
        
        # 設定 hover meta（讓你移到點上可看到座標/ts）
        # 對 pred 小點
        meta_pred = []
        for i, p in enumerate(pred):
            meta_pred.append({
                'x': p['pos'].get('x'), 'y': p['pos'].get('y'), 'z': p['pos'].get('z'),
                'fid': p.get('id'), 'ts': p.get('timestamp'), 'src': 'pred-traj'
            })
        scat_pred.meta = meta_pred

        # 對落地點
        scat_landing.meta = [{'x': lpos.get('x'), 'y': ly, 'z': lz, 'fid': end_fid, 'ts': lt, 'src': 'pred-landing'}]

        # Rally 過濾（跟 BridgeGap 一樣，先標記 rally_id，之後用 _refresh_... 控制顯示）
        r_id = self.current_rally_id or 0
        for art in (line_pred, scat_pred, scat_landing):
            art.rally_id = r_id
            self._landing_artists.append(art)

        # 存 session 用
        self.landings.append({
            'rally_id': r_id,
            'pred': [
                {
                    'pos': {'x': p['pos'].get('x'), 'y': p['pos'].get('y'), 'z': p['pos'].get('z')},
                    'id':  p.get('id'),
                    'timestamp': p.get('timestamp')
                } for p in pred
            ],
            'landing': {
                'pos': {'x': lpos.get('x'), 'y': ly, 'z': lz},
                'id':  end_fid,
                'timestamp': lt
            }
        })

        # 立即重繪
        self.new_data_received = True

    def process_balltype_input_message(self, msg):
        """
        顯示 points_for_balltype（以 y,z 作圖；含 fid/ts 做 hover）。
        支援兩種 payload：
        A) {"linear":[{"pos":{"x":..,"y":..,"z":..},"id":..,"timestamp":..}, ...]}
        B) {"points_for_balltype":[同上]}  # 若你放進 Segment/Debug 的副欄位
        """
        payload = json.loads(msg.payload)

        linear = payload.get('linear')
        if linear is None:
            linear = payload.get('points_for_balltype', [])  # 兼容副欄位名稱

        if not linear:
            return

        pts, meta = [], []
        for p in linear:
            pos = p.get('pos', {})
            y = pos.get('y', None)
            z = pos.get('z', None)
            if y is None or z is None:
                continue
            pts.append((y, z))
            meta.append({
                'x': pos.get('x'), 'y': y, 'z': z,
                'fid': p.get('id'), 'ts': p.get('timestamp'),
                'src': 'balltype'
            })

        r_id = self.current_rally_id or 0
        self.balltype_inputs.append({'points': pts, 'meta': meta, 'rally_id': r_id})

        self.new_data_received = True  # 觸發重繪
    
    def _refresh_bridgegap_visibility(self):
        """依 self.selected_rally_id 決定每個 bridge artist 要不要顯示 / 淡化。"""
        if not self.show_bridgegap:
            # 直接把全部 BridgeGap 物件隱藏
            for art in self._bridge_artists:
                art.set_visible(False)
            return
        
        sel = self.selected_rally_id or 0
        for art in self._bridge_artists:
            r_id = getattr(art, "rally_id", 0)     # 沒標就當 0 = 全域
            # 👉 若選了特定 Rally，則顯示 (r_id == 該場) 或 (r_id == 0 代表全域)
            visible = True if sel == 0 else (r_id == sel or r_id == 0)
            art.set_visible(visible)

    def _toggle_bridgegap(self, event):
        # ① 切換狀態
        self.show_bridgegap = not self.show_bridgegap
        # ② 更新按鈕文字
        self.btn_bg.label.set_text('BridgeGap: ON' if self.show_bridgegap else 'BridgeGap: OFF')
        # ③ 即時刷新可見度
        self._refresh_bridgegap_visibility()
        self.fig.canvas.draw_idle()

    def _refresh_balltype_visibility(self):
        """
        依 show_balltype 與 selected_rally_id 控制 BallTypeInput 顯示。
        self.balltype_inputs: [{'points':[(y,z),...], 'meta':[...], 'rally_id':int}, ...]
        """
        if not hasattr(self, 'scat_balltype'):
            return

        if not getattr(self, 'show_balltype', True):
            # 關閉：清空資料
            self.scat_balltype.set_offsets(np.empty((0,2)))
            self.scat_balltype.meta = []
            return

        # 開啟：整理目前應顯示的點（依 Rally 過濾）
        bt_yz, bt_meta = [], []
        sel = getattr(self, 'selected_rally_id', 0) or 0
        for item in getattr(self, 'balltype_inputs', []):
            r_id = item.get('rally_id', 0)
            if sel and r_id != sel:
                continue
            bt_yz.extend(item.get('points', []))
            bt_meta.extend(item.get('meta', []))

        if bt_yz:
            self.scat_balltype.set_offsets(bt_yz)
            # 統一 meta 結構，供 hover 用
            self.scat_balltype.meta = [
                {'x': m.get('x'), 'y': m.get('y'), 'z': m.get('z'),
                'fid': m.get('fid'), 'ts': m.get('ts'), 'src': 'balltype'}
                for m in bt_meta
            ]
        else:
            self.scat_balltype.set_offsets(np.empty((0,2)))
            self.scat_balltype.meta = []

    def _toggle_balltype(self, event):
        self.show_balltype = not self.show_balltype
        self.btn_balltype.label.set_text('BallType: ON' if self.show_balltype else 'BallType: OFF')
        # 立刻刷新顯示
        self._refresh_balltype_visibility()
        self.fig.canvas.draw_idle()


    def _refresh_landing_visibility(self):
        """依 self.selected_rally_id 決定預測軌跡/落地點要不要顯示。"""
        for art in self._landing_artists:
            r_id = getattr(art, "rally_id", 0)
            if self.selected_rally_id and r_id != self.selected_rally_id:
                art.set_visible(False)
            else:
                art.set_visible(True)
    
    
    def update_text_display(self):
        try:
            if self.ax_text is None or self.fig is None:
                return
            self.ax_text.cla()
            self.ax_text.axis('off')

            max_follow_lines = 35
            use_follow_latest = getattr(self, "follow_latest", True)
            total_lines = len(self.text_lines)

            if use_follow_latest:
                visible_lines = self.text_lines[-max_follow_lines:]
                start = len(self.text_lines) - len(visible_lines)
            else:
                start = self.text_offset
                end = start + self.text_display_count
                visible_lines = self.text_lines[start:end]

            # if use_follow_latest:
            #     start = max(0, total_lines - max_follow_lines)
            # else:
            #     start = self.text_offset
            # end = min(start + self.text_display_count, total_lines)
            # visible_lines = self.text_lines[start:end]

            # segment_lines = 0
            # for line in visible_lines:
            #     if 'Segment' in line:
            #         segment_lines += 1

            segment_lines = sum(1 for line in visible_lines if "Segment" in line)

            status_text = f"Total: {total_lines},   Segment: {segment_lines}"
            self.ax_text.text(0.01, 1.01, status_text, fontsize=10, color='gray', ha='left', va='bottom')
            
            highlight_start = highlight_end = None
            if self.selected_rally_id:
                tag_start = f"Rally {self.selected_rally_id}"
                # 起點: 第一條包含 "Rally {self.selected_rally_id}" 的行
                for i, ln in enumerate(self.text_lines):
                    if tag_start in ln:
                        highlight_start = i
                        # print(f"Highlight start at line {highlight_start} for rally {self.selected_rally_id}")
                        break
                # 終點: 起點之後第一條只含 "-" 分隔線的行
                if highlight_start is not None:
                    for j in range(highlight_start + 1, len(self.text_lines)):
                        if set(self.text_lines[j]) == {"-"}:
                            highlight_end = j
                            # print(f"Highlight end at line {highlight_end} for rally {self.selected_rally_id}")
                            break
                        # if self.text_lines[j].startswith('------------------------------------------'):
                        #     highlight_end = j
                        #     break
                    if highlight_end is None:
                        highlight_end = len(self.text_lines) - 1  # 如果沒有找到結束行，就設為最後一行
            
            
            bbox = dict(boxstyle='round,pad=0.15', fc='#FFF9CC', ec='none', alpha=0.5)
            for i, line in enumerate(visible_lines):
                line_number = start + i + 1
                line = f"{line_number:>4}:    {line}"

                # highlight = self.selected_rally_id and f"R{self.selected_rally_id}" in line
                global_idx = (total_lines - len(visible_lines) + i) if self.follow_latest else (self.text_offset + i)
                highlight = (highlight_start is not None and 
                             highlight_start <= global_idx <= highlight_end)

                if "Event" in line:
                    if "Serve" in line:
                        color = 'red'
                    elif "Hit" in line:
                        color = 'blue'
                    elif "Dead" in line:
                        color = 'green'
                    else:
                        color = 'black'
                elif "Segment" in line:
                    color = 'black'
                else:
                    color = 'black'

                # self.ax_text.text(
                #     0.01, 1 - i * 0.03, line, fontsize=10, color=color, bbox=bbox if highlight else None, ha='left', va='top'
                # )
                y = 1 - i * self.line_step
                txt = self.ax_text.text(
                    0.01, 1 - i * 0.03, line, fontsize=10, color=color, bbox=bbox if highlight else None,
                    ha='left', va='top', picker=True                # ← ① 讓文字可點
                )

                # 這個 global_idx 已經算好了：不要覆寫它
                global_idx_for_text = start + i  # 用新的變數名，避免覆寫上面的 global_idx

                # ServeCheck 行樣式（淡底色、不會太飽和）
                if hasattr(self, 'line_idx_to_servecheck') and (global_idx_for_text in self.line_idx_to_servecheck):
                    sc_idx = self.line_idx_to_servecheck[global_idx_for_text]
                    txt.servecheck_idx = sc_idx
                    ok_flag = self.serve_checks[sc_idx]['ok']

                    # 白字 + 淡底色
                    txt.set_color("white")
                    txt.set_backgroundcolor(mcolors.to_rgba("tab:red" if ok_flag else "tab:gray", alpha=0.5))

                    # 是否為被選中那一行
                    is_selected = (getattr(self, 'selected_servecheck_idx', None) == sc_idx)
                    if is_selected:
                        txt.set_fontweight("bold")

                        # 把背景色存在變數
                        bg = mcolors.to_rgba("tab:red" if ok_flag else "tab:gray", alpha=0.5)

                        # 先設背景色
                        txt.set_backgroundcolor(bg)

                        # bbox 用同樣的顏色
                        txt.set_bbox(dict(
                            facecolor=bg,
                            edgecolor="yellow",
                            linewidth=1.2,
                            boxstyle="round,pad=0.10"
                        ))
                        txt.set_zorder(4)   # 被選中行（含 bbox）放低一點
                    else:
                        txt.set_zorder(5)   # 其他行放高一點，確保蓋在選中行的 bbox 上
                else:
                    # 非 ServeCheck 行
                    txt.set_zorder(6)       # 一般行再高一點，保險
                    # 保留原本被 Rally 範圍高亮的 bbox，但縮小 padding
                    if highlight:
                        txt.set_bbox(dict(boxstyle='round,pad=0.10', fc='#FFF9CC', ec='none', alpha=0.7))

                # ② 若這行對應某個 segment，就存進文字物件
                # global_idx = start + i        # 這行在 self.text_lines 的實際索引
                if global_idx_for_text in self.line_idx_to_segment:
                    txt.segment_id = self.line_idx_to_segment[global_idx_for_text]

                seg_attr = getattr(txt, 'segment_id', None)
                if self.selected_segment_id is not None and seg_attr == self.selected_segment_id:
                    txt.set_fontweight('bold')

            self.fig.canvas.draw_idle()

        except Exception as e:
            logging.error(f"Error updating text display: {e}")

    
    def update_rally_buttons(self):
        """Rebuild RadioButtons when a new rally is detected."""
        labels = ['All'] + [f'R{rid}' for rid in range(1, self.current_rally_id + 1)]
        # remove existing axes & widget
        self.ax_rally.cla()
        self.ax_rally.remove()
        self.ax_rally = self.fig.add_axes([0.66, 0.1, 0.05, 0.8])
        # self.ax_rally.set_anchor('NW') 
        self.rally_radio = RadioButtons(self.ax_rally, labels, active=0)
        self.rally_radio.on_clicked(self.on_rally_selected)
        self.fig.canvas.draw_idle()

    def on_rally_selected(self, label):
        # self.selected_rally_id = 0 if label == 'All' else int(label[1:])
        # self.update_plot()
        # self.update_text_display()

        cur_idx = 0 if self.selected_rally_id == 0 else self.selected_rally_id
        clicked_idx = 0 if label == 'All' else int(label[1:])

        if clicked_idx == cur_idx and clicked_idx != 0:
            self.rally_radio.set_active(0)  # Reset to "All" if the same rally is clicked
            self.selected_rally_id = 0
        else:
            self.selected_rally_id = clicked_idx
            # self.rally_radio.set_active(clicked_idx)
            if self.selected_rally_id == 0:
                self.follow_latest = True
            else:
                tag = f'Rally {self.selected_rally_id}'
                try:
                    idx = next(i for i, ln in enumerate(self.text_lines) if tag in ln)
                    # print(f"Selected rally {self.selected_rally_id} at index {idx}")
                except StopIteration:
                    idx = 0
                self.follow_latest = False
                # self.text_offset = max(0, idx - self.text_display_count + 1)
                self.text_offset = idx
                # print(f"Setting text offset to {self.text_offset} for rally {self.selected_rally_id}")

        self.update_plot()
        self.update_text_display()
    
    
    def on_pick_segment(self, event):
        art = event.artist

        # ---- ServeCheck 物件 ----
        sc_idx = getattr(art, 'servecheck_idx', None)
        if sc_idx is not None:
            # toggle：再次點同一行就取消
            if self.selected_servecheck_idx == sc_idx:
                self._clear_servecheck_highlight()
                self.selected_servecheck_idx = None
                self.update_text_display()
                return

            # 換選別的 → 先清舊 highlight，再畫新 highlight
            self._clear_servecheck_highlight()
            self.selected_servecheck_idx = sc_idx
            self._draw_servecheck_highlight(sc_idx)
            self.update_text_display()
            return


        # ---- Segment 物件 ----
        if not hasattr(art, 'segment_id'):
            return

        seg_id = art.segment_id

        # ◼ 若已選中同一條 → 取消高亮
        if self.selected_segment_id == seg_id:
            self._clear_segment_highlight()
            self.selected_segment_id = None

            self.update_text_display()
            return
            

        # ◼ 切換到新 Segment
        self.selected_segment_id = seg_id
        self.update_text_display()  # 更新文字顯示

        # 如果點到的是文字，先找對應的 scatter
        if isinstance(art, mtext.Text):          # or: if not hasattr(art, "get_offsets")
            art = self._find_scatter_by_segment(seg_id)
            if art is None:
                return                          # 極端情況：還沒有任何 scatter 被畫出

        self._draw_segment_highlight(art)

    def _clear_segment_highlight(self):
        # 把舊的高亮刪掉
        if self.highlight_artist is not None:
            try:
                self.highlight_artist.remove()
            except ValueError:
                pass
            self.highlight_artist = None
            self.fig.canvas.draw_idle()

    def _draw_segment_highlight(self, base_art):
        self._clear_segment_highlight()

        xy = base_art.get_offsets()
        # 全部點用同一顏色，尺寸較大、加邊線
        self.highlight_artist = self.ax.scatter(
            xy[:, 0], xy[:, 1],
            s=30,            # 比原本 5 大很多
            c='none',        # 填色透明
            edgecolors='yellow',
            linewidths=1.5,
            zorder=3
        )
        self.fig.canvas.draw_idle()

    def _find_scatter_by_segment(self, seg_id):
        """回傳該 seg_id 對應的 scatter artist；找不到就傳 None"""
        for art in self.segment_artists:
            if getattr(art, 'segment_id', None) == seg_id:
                return art
        return None
    
    
    def _clear_servecheck_highlight(self):
        for a in getattr(self, 'highlight_sc_artists', []):
            try: a.remove()
            except Exception: pass
        self.highlight_sc_artists.clear()
        if self.fig:
            self.fig.canvas.draw_idle()

    def _draw_servecheck_highlight(self, sc_idx):
        sc = self.serve_checks[sc_idx]
        pts = sc.get('points', [])
        if not pts:
            return
        yy, zz = zip(*pts)
        hl = self.ax.scatter(yy, zz, s=60, facecolors='none', edgecolors='red',
                            linewidths=1.5, zorder=10)
        self.highlight_sc_artists.append(hl)
        if self.fig:
            self.fig.canvas.draw_idle()

    def _rebuild_servecheck_line_map(self):
        """
        根據 self.text_lines 中出現的 ServeCheck 行，把其行索引對回 self.serve_checks 的順序索引。
        假設你寫入 text_lines 的邏輯是：每新增一筆 ServeCheck，就 append 一行
        f"[ServeCheck OK/NG] {start_fid} - {end_fid}" —— 和目前程式一致。
        """
        self.line_idx_to_servecheck = {}
        sc_idx = 0
        for i, ln in enumerate(self.text_lines):
            if ln.startswith("[ServeCheck "):  # "[ServeCheck OK]" 或 "[ServeCheck NG]"
                if sc_idx < len(self.serve_checks):
                    self.line_idx_to_servecheck[i] = sc_idx
                    sc_idx += 1
                else:
                    # 若文本比 serve_checks 多，保守忽略多出來的行
                    break

    
    def clear_highlight(self):
        for art in self.highlight_artists:
            art.remove()
        self.highlight_artists.clear()

    
    def update_plot(self):
        # Clear previous segment artists so we can redraw filtered view
        for art in self.segment_artists:
            art.remove()
        self.segment_artists.clear()

        for txt in self.event_texts:
            txt.remove()
        self.event_texts.clear()

        # Update normal points ("point")
        point_data = (
            list(self.point_buffers['point'].queue) if self.max_points else self.point_buffers['point']
        )
        if point_data:
            # point_data: list of (x, y, z, fid, ts)
            y_data = [it[1] for it in point_data]
            z_data = [it[2] for it in point_data]
            self.scat_point.set_offsets(list(zip(y_data, z_data)))
            self.scat_point.meta = [
                {'x': it[0], 'y': it[1], 'z': it[2], 'fid': it[3], 'ts': it[4], 'src': 'point'}
                for it in point_data
            ]
        else:
            self.scat_point.set_offsets(np.empty((0, 2)))
            self.scat_point.meta = []

        # Update event points by type
        self.scat_event_hit.set_offsets(np.empty((0, 2)))
        self.scat_event_hit.meta = []
        self.scat_event_serve.set_offsets(np.empty((0, 2)))
        self.scat_event_serve.meta = []
        self.scat_event_dead.set_offsets(np.empty((0, 2)))
        self.scat_event_dead.meta = []
        event_map = {
            1: {'scatter': self.scat_event_hit, 'color': 'blue'},
            2: {'scatter': self.scat_event_serve, 'color': 'red'},
            3: {'scatter': self.scat_event_dead, 'color': 'green'},
        }

        if self.office:
            y_offset = 0.2
        else:
            y_offset = 0.5

        # ───────────────────────── Events, filtered ──────────────────────
        for event_type , buffer in self.event_buffers.items():
            data = list(buffer.queue) # (x, y, z, fid, rally_id, ts)
            if not data:
                continue
            if self.selected_rally_id:
                data = [d for d in data if d[4] == self.selected_rally_id]
            if data:
                y_data = [d[1] for d in data]
                z_data = [d[2] for d in data]
                art = event_map[event_type]['scatter']
                art.set_offsets(list(zip(y_data, z_data)))
                art.meta = [
                    {'x': d[0], 'y': d[1], 'z': d[2], 'fid': d[3], 'ts': d[5], 'rally_id': d[4], 'src': f'event-{event_type}'}
                    for d in data
                ]

                for x, y, z, fid, r_id, ts in data:
                    t = self.ax.text((y-y_offset) if y < 0 else y, z - 0.25, f"{fid} (R{r_id})",
                                     fontsize=9, color=event_map[event_type]['color'],
                                     ha='left', va='bottom')
                    self.event_texts.append(t)

        # ───────────────────────── Segments, filtered ────────────────────
        for seg in self.segment_data:
            if self.selected_rally_id and seg['rally_id'] != self.selected_rally_id:
                # print(f"-- Skipping segment {seg['segment_id']} for rally {self.selected_rally_id}")
                continue
            # print(f"++ Drawing segment {seg['segment_id']} for rally {seg['rally_id']}")
            y_d, z_d = zip(*seg['points'])
            n = len(y_d)
            base_rgb = seg['color']
            h, s, v = colorsys.rgb_to_hsv(*base_rgb)
            v_vals = np.linspace(1, 0.3, n)
            grad_colors = [colorsys.hsv_to_rgb(h, s, v_val) for v_val in v_vals]
            art = self.ax.scatter(y_d, z_d, s=5, c=grad_colors, picker=5)
            art.segment_id = seg['segment_id']
            art.meta = seg.get('meta', [])  # list of {'x','y','z','fid','ts'}
            self.segment_artists.append(art)

        # ───────────────────────── ServeCheck 視覺化 ────────────────────
        ng_yz = []
        ng_meta = []  # for hover
        for sc in self.serve_checks:
            if self.selected_rally_id and sc['rally_id'] != self.selected_rally_id:
                continue
            if not sc['ok']:
                ng_yz.extend(sc['points'])
                # sc['meta'] 對應 linear 每點，含 fid/ts（在 process_servecheck_message 會存）
                ng_meta.extend(sc.get('meta', [{'y':y,'z':z,'fid':None,'ts':None} for (y,z) in sc['points']]))

        if ng_yz:
            self.scat_serve_ng.set_offsets(ng_yz)
            # 轉成 hover 用的 meta 結構（y,z 必要；x 不用也可）
            self.scat_serve_ng.meta = [
                {'x': None, 'y': m.get('y'), 'z': m.get('z'), 'fid': m.get('fid'), 'ts': m.get('ts'), 'src': 'servecheck-ng'}
                for m in ng_meta
            ]
        else:
            self.scat_serve_ng.set_offsets(np.empty((0,2)))
            self.scat_serve_ng.meta = []

        # ───────────────────────── BridgeGap 視覺化 ──────────────────────        
        self._refresh_bridgegap_visibility()

        # ───────────────────────── Pred Landing 視覺化 ──────────────────────    
        self._refresh_landing_visibility()

        # # ───────────────────────── BallTypeInput 視覺化 ────────────────────
        # bt_yz, bt_meta = [], []
        # for bt in getattr(self, 'balltype_inputs', []):
        #     if self.selected_rally_id and bt['rally_id'] != self.selected_rally_id:
        #         continue
        #     bt_yz.extend(bt['points'])
        #     bt_meta.extend(bt.get('meta', []))

        # if bt_yz:
        #     self.scat_balltype.set_offsets(bt_yz)
        #     # hover meta（結構與其他 scatter 一致）
        #     self.scat_balltype.meta = [
        #         {'x': m.get('x'), 'y': m.get('y'), 'z': m.get('z'),
        #         'fid': m.get('fid'), 'ts': m.get('ts'), 'src': 'balltype'}
        #         for m in bt_meta
        #     ]
        # else:
        #     self.scat_balltype.set_offsets(np.empty((0,2)))
        #     self.scat_balltype.meta = []

        # ───────────────────────── BallTypeInput 視覺化 ────────────────────
        if hasattr(self, 'scat_balltype'):
            self._refresh_balltype_visibility()

        if self.fig is None:
            print("Warning: Figure is not initialized!")
            return
        
        try:
            self.fig.canvas.draw_idle()  # Update the plot
            # self.fig.canvas.flush_events()
        except Exception as e:
            logging.error(f"Error during draw: {e}. Resetting ticks.")
            self.ax.set_xticks(range(self.x_min, self.x_max+1))
            self.ax.set_yticks(range(self.y_min, self.y_max+1))

        self._update_fid_highlight()

     
    def on_key_press(self, event):
        if event.key == "c":  # 按下 c 鍵
            self.clear_data()

        # 按下 ESC 取消 fid 搜尋高亮並清空輸入框、取消 Segment 高亮
        if event.key in ("escape", "esc"):
            # if hasattr(self, "fid_tb"):
            #     self.clear_highlight()
            #     self.selected_servecheck_idx = None
            #     self.fig.canvas.draw_idle()

            # 移除 ServeCheck highlight
            if getattr(self, "selected_servecheck_idx", None) is not None:
                self._clear_servecheck_highlight()
                self.selected_servecheck_idx = None
            
            # 移除高亮
            if hasattr(self, "_clear_fid_highlight"):
                self._clear_fid_highlight()

            # 清除查詢狀態與輸入框
            self._fid_query = None
            if hasattr(self, "fid_tb"):
                try:
                    self.fid_tb.set_val("")
                except Exception:
                    pass

            # 移除 Segment 高亮
            if getattr(self, "selected_segment_id", None) is not None:
                if hasattr(self, "_clear_segment_highlight"):
                    self._clear_segment_highlight()
                self.selected_segment_id = None
                self.update_text_display()

            if self.fig:
                self.fig.canvas.draw_idle()
            return


    def on_scroll(self, event):
        # 初始化 follow_latest 變數
        if not hasattr(self, "follow_latest"):
            self.follow_latest = True

        if self.follow_latest:
            if event.button == 'up':
                # 滾輪往上時退出追蹤模式
                self.follow_latest = False
                self.text_offset = max(0, len(self.text_lines) - self.text_display_count - 1)
        else:
            if event.button == 'up':
                self.text_offset = max(0, self.text_offset - 1)
            elif event.button == 'down':
                max_offset = max(0, len(self.text_lines) - self.text_display_count)
                self.text_offset = min(max_offset, self.text_offset + 1)

                # 滾到底部時切換成 follow_latest 模式
                if self.text_offset + self.text_display_count >= len(self.text_lines):
                    self.follow_latest = True

        self.update_text_display()


    def on_go_bottom_clicked(self, event):
        self.follow_latest = True
        self.update_text_display()


    def _apply_hover_style(self, kind: str):
        """
        根據種類套用 hover_annot 的樣式。
        kind = 'landing' 使用綠色樣式；其它一律回到預設白底樣式。
        """
        # 背景框
        box = self.hover_annot.get_bbox_patch()
        # 箭頭
        ap = getattr(self.hover_annot, "arrow_patch", None)

        if kind == 'landing':
            # 綠色系：醒目表示落地點
            box.set_facecolor("#E9FFE9")
            box.set_edgecolor("#2E7D32")
            box.set_alpha(0.98)
            if ap is not None:
                ap.set_edgecolor("#2E7D32")
                ap.set_facecolor("#2E7D32")
                ap.set_linewidth(0.9)
                ap.set_alpha(0.9)
        else:
            # 回復預設白底
            box.set_facecolor("white")
            box.set_edgecolor("0.7")
            box.set_alpha(0.95)
            if ap is not None:
                ap.set_edgecolor("0.3")
                ap.set_facecolor("0.3")
                ap.set_linewidth(0.5)
                ap.set_alpha(0.8)
    
    def on_hover(self, event):
        # # 只在主座標軸內有效
        # if event.inaxes != self.ax:
        #     if self.hover_annot.get_visible():
        #         self.hover_annot.set_visible(False)
        #         self.fig.canvas.draw_idle()
        #     return
        # 允許 ax 或 ax_overlay 都能觸發
        if event.inaxes not in (self.ax, getattr(self, 'ax_overlay', None)):
            if self.hover_annot.get_visible():
                self.hover_annot.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        def _fmt(v):
            try:
                return f"{float(v):.3f}"
            except Exception:
                return str(v)

        # 依優先序檢查的 artists
        candidate_arts = []
        candidate_arts.extend(self.segment_artists)
        candidate_arts.extend([self.scat_event_hit, self.scat_event_serve, self.scat_event_dead])
        candidate_arts.append(self.scat_point)
        candidate_arts.append(self.scat_serve_ng)
        if hasattr(self, 'scat_balltype'):
            candidate_arts.append(self.scat_balltype)
        candidate_arts.extend(getattr(self, "_landing_artists", []))

        hit_found = False
        for art in candidate_arts:
            if art is None:
                continue

            # 跳過沒有 contains 的（例如 Line2D）
            if not hasattr(art, "contains"):
                continue

            contains, info = art.contains(event)
            if not contains:
                continue

            inds = info.get('ind', None)
            if inds is None:
                continue
            inds = np.atleast_1d(inds)
            if inds.size == 0:
                continue

            idx = int(inds[0])

            # 取 meta
            meta = getattr(art, 'meta', None)
            if not meta or idx >= len(meta):
                continue

            m   = meta[idx]
            x   = m.get('x')
            y   = m.get('y')
            z   = m.get('z')
            fid = m.get('fid', 'N/A')
            ts  = m.get('ts', None)
            src = m.get('src', '')

            # timestamp 文字
            ts_str = "-"
            try:
                if ts is not None:
                    ts_str = f"{ts:.3f}s"
            except Exception:
                ts_str = str(ts)

            # Hover 文字：Landing 用不同抬頭
            if src == 'pred-landing':
                text = f"★ Landing\nfid: {fid}\nxyz: ({_fmt(x)}, {_fmt(y)}, {_fmt(z)})\nts: {ts_str}"
            else:
                text = f"fid: {fid}\nxyz: ({_fmt(x)}, {_fmt(y)}, {_fmt(z)})\nts: {ts_str}"

            # 取得該點的 (y,z)（注意圖上是 y,z）
            yy, zz = event.xdata, event.ydata
            if hasattr(art, "get_offsets"):
                offs = art.get_offsets()
                try:
                    if idx < len(offs):
                        yy, zz = offs[idx]
                except Exception:
                    pass

            # 將 annotation 錨在該點
            self.hover_annot.xy = (yy, zz)
            self.hover_annot.set_text(text)

            # ── 左右方向切換 ──
            xlim = self.ax.get_xlim()
            midx = 0.5 * (xlim[0] + xlim[1])

            # if yy >= midx:
            #     # 右半邊 → 往左擺，錨在文字框右邊界
            #     xoff, yoff = -12, 12
            #     self.hover_annot.set_ha('right')
            #     self.hover_annot.set_va('bottom')
            # else:
            #     # 左半邊 → 往右擺，錨在文字框左邊界
            #     xoff, yoff = 12, 12
            #     self.hover_annot.set_ha('left')
            #     self.hover_annot.set_va('bottom')

            # # 三重保險更新偏移（不同後端一致）
            # self.hover_annot.set_position((xoff, yoff))
            # self.hover_annot.xyann  = (xoff, yoff)
            # self.hover_annot.xytext = (xoff, yoff)
            # self.hover_annot.set_zorder(999)

            # Landing 用綠色樣式，其它回復預設樣式
            if src == 'pred-landing' or src == 'pred-traj':
                self._apply_hover_style('landing')
            else:
                self._apply_hover_style('default')

            self.hover_annot.set_visible(True)
            self.fig.canvas.draw_idle()
            hit_found = True
            break

        if not hit_found and self.hover_annot.get_visible():
            self.hover_annot.set_visible(False)
            self.fig.canvas.draw_idle()


    def _on_save_session(self, event=None):
        # 1) 當下時間作為預設名
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 2) 優先用輸入框文字，否則用最新 log 資料夾名稱，再否則用當下時間
        raw = (self.save_name_tb.text or "").strip()
        filename = raw or self._get_latest_log_name() or now_str

        # 3) 執行存檔
        self._do_save_session(filename)

        # 4) 更新預設檔名（之後再存會用這個）並清空輸入框
        self.default_save_name = filename
        self.save_name_tb.set_val("")

    def _do_save_session(self, filename: str):
        # 建目錄：./drawRecord/<filename>/
        base_dir = os.path.join(ROOTDIR, "LayerContent", "drawRecord")
        os.makedirs(base_dir, exist_ok=True)
        sess_dir = os.path.join(base_dir, filename)
        os.makedirs(sess_dir, exist_ok=True)

        # 存 PNG（當前畫面）
        png_path = os.path.join(sess_dir, f"{filename}_fig.png")
        self.fig.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Screenshot: {png_path}")

        # （可選）若你已經有 _collect_session_state()，順便存 JSON 快照
        if hasattr(self, "_collect_session_state") and callable(self._collect_session_state):
            snap = self._collect_session_state()
            json_path = os.path.join(sess_dir, f"{filename}_session.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
            print(f"[Saved] Session JSON: {json_path}")
        else:
            print("[Info] No _collect_session_state() found. Saved PNG only.")

    def _get_latest_log_name(self):
        log_dir = os.path.join(ROOTDIR, "log")
        if not os.path.isdir(log_dir):
            return None
        dirs = [d for d in os.listdir(log_dir) 
                if os.path.isdir(os.path.join(log_dir, d))]
        if not dirs:
            return None
        latest = max(dirs, key=lambda d: os.path.getmtime(os.path.join(log_dir, d)))
        return latest  # 資料夾名稱本身，例如 "20250810_173237"
    
    
    def _queue_to_list(self, q_or_list):
        if isinstance(q_or_list, queue.Queue):
            return list(q_or_list.queue)
        return list(q_or_list)
    
    def _collect_session_state(self) -> dict:
        # points
        points = self._queue_to_list(self.point_buffers['point'])
        # events
        events = {
            str(k): self._queue_to_list(buf)
            for k, buf in self.event_buffers.items()
        }
        # segments（已是 list[dict]，但顏色 tuple 要確保可序列化）
        segs = []
        for seg in self.segment_data:
            segs.append({
                "segment_id": seg["segment_id"],
                "points": seg["points"],         # list of (y,z)
                "fids": seg["fids"],
                "meta": seg["meta"],             # list of {x,y,z,fid,ts}
                "color": list(seg["color"]),     # 轉 list
                "rally_id": seg["rally_id"],
            })

        state = {
            "meta": {
                "app_uuid": self.app_uuid,
                "timestamp": self.timestamp,
                "office": self.office,
                "axis": {
                    "xlim": [self.x_min, self.x_max],
                    "ylim": [self.y_min, self.y_max],
                }
            },
            "rally": {
                "current_rally_id": self.current_rally_id,
                "selected_rally_id": self.selected_rally_id,
            },
            "selection": {
                "selected_segment_id": self.selected_segment_id
            },
            "text_panel": {
                "text_lines": self.text_lines,
                "line_idx_to_segment": self.line_idx_to_segment,
                "follow_latest": getattr(self, "follow_latest", True),
                "text_offset": getattr(self, "text_offset", 0),
                "text_display_count": getattr(self, "text_display_count", 35),
            },
            "buffers": {
                "points": points,           # [(x,y,z,fid,ts), ...]
                "events": events,           # {"1": [...], "2": [...], "3": [...]}
            },
            "segments": segs,
            "bridge_gaps": self.bridge_gaps,
            "ui": {
                "show_bridgegap": self.show_bridgegap,
                "show_balltype": getattr(self, "show_balltype", True)
            },
            "serve_checks": self.serve_checks,
            "serve_ui": {
                "selected_servecheck_idx": self.selected_servecheck_idx
            },
            "landings": self.landings,
            "balltype_inputs": [
                {
                    "rally_id": item.get("rally_id", 0),
                    "points":   [(float(y), float(z)) for (y, z) in item.get("points", [])],
                    "meta":     [
                        {
                            "x": m.get("x"), "y": m.get("y"), "z": m.get("z"),
                            "fid": m.get("fid"), "ts": m.get("ts")
                        } for m in item.get("meta", [])
                    ]
                }
                for item in getattr(self, "balltype_inputs", [])
            ],
        }
        return state
    
    def _load_session(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            snap = json.load(f)

        # 軸設定
        axis = snap["meta"]["axis"]
        self.x_min, self.x_max = axis["xlim"]
        self.y_min, self.y_max = axis["ylim"]
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_xticks(range(self.x_min, self.x_max+1))
        self.ax.set_yticks(range(self.y_min, self.y_max+1))

        # Rally
        self.current_rally_id = snap["rally"]["current_rally_id"]
        self.selected_rally_id = snap["rally"]["selected_rally_id"]

        # Text 面板
        tp = snap["text_panel"]
        self.text_lines = tp["text_lines"]
        self.line_idx_to_segment = {int(k): v for k, v in tp["line_idx_to_segment"].items()} if isinstance(tp["line_idx_to_segment"], dict) else tp["line_idx_to_segment"]
        self.follow_latest = tp["follow_latest"]
        self.text_offset = tp["text_offset"]
        self.text_display_count = tp["text_display_count"]

        # Buffers（轉回 Queue 或 list）
        self.point_buffers = {
            'point': queue.Queue(maxsize=self.max_points) if self.max_points else [],
            'event': queue.Queue(maxsize=self.max_points) if self.max_points else [],
        }
        pts = snap["buffers"]["points"]
        if self.max_points:
            for it in pts:
                if self.point_buffers['point'].full(): self.point_buffers['point'].get()
                self.point_buffers['point'].put(tuple(it))
        else:
            self.point_buffers['point'] = [tuple(it) for it in pts]

        # events
        self.event_buffers = {
            1: queue.Queue(maxsize=self.max_points or 0),
            2: queue.Queue(maxsize=self.max_points or 0),
            3: queue.Queue(maxsize=self.max_points or 0),
        }
        for k in ["1","2","3"]:
            for it in snap["buffers"]["events"].get(k, []):
                self.event_buffers[int(k)].put(tuple(it))

        # segments
        self.segment_data.clear()
        for seg in snap["segments"]:
            seg = dict(seg)
            seg["color"] = tuple(seg["color"])
            self.segment_data.append(seg)

        self.selected_segment_id = snap["selection"]["selected_segment_id"]
        
        # --- BridgeGap ---
        for art in getattr(self, "_bridge_artists", []):
            try: art.remove()
            except Exception: pass
        self._bridge_artists = []

        # 讀回資料
        self.bridge_gaps = snap.get("bridge_gaps", [])

        # 幫助函式：用快照畫出一筆 BridgeGap
        def _draw_bridgegap_from_snap(bg):
            objs = []
            def plot_line(seq, color):
                if not seq: return
                yy = [p['y'] for p in seq if p.get('y') is not None]
                zz = [p['z'] for p in seq if p.get('z') is not None]
                if not yy or not zz: return
                line, = self.ax.plot(yy, zz, '-o', color=color, markersize=3, linewidth=0.8)
                objs.append(line)

            plot_line(bg.get('fwd', []), 'cyan')
            plot_line(bg.get('bwd', []), 'magenta')

            def plot_point(p, marker='o', s=40):
                if not p: return
                y, z = p.get('y'), p.get('z')
                if y is None or z is None: return
                sc = self.ax.scatter([y], [z], s=s, c='orange', marker=marker, zorder=16)
                objs.append(sc)
                # 擊球點加時間標
                if marker == '*' and p.get('t') is not None:
                    txt = self.ax.text(y, z - 0.2, f"{p['t']:.3f}", fontsize=9, color='orange',
                                    ha='center', va='top')
                    objs.append(txt)

            plot_point(bg.get('p_fwd'), 'o', 40)
            plot_point(bg.get('p_bwd'), 'o', 40)
            plot_point(bg.get('hit'),   '*', 80)

            # 標註 rally_id+zorder 並收進集中管理
            r_id = bg.get('rally_id', 0)
            for a in objs:
                try: a.set_zorder(15)
                except Exception: pass
                setattr(a, 'rally_id', r_id)
                self._bridge_artists.append(a)

        # 逐筆重建
        for bg in self.bridge_gaps:
            _draw_bridgegap_from_snap(bg)

        # 套用可見度（依 show_bridgegap 與 selected_rally_id）
        self._refresh_bridgegap_visibility()

        # --- ServeCheck ---
        self.serve_checks = snap.get("serve_checks", [])
        # 載入時不直接用舊的選取，先取消（避免索引對不上）
        self.selected_servecheck_idx = None
        self._clear_servecheck_highlight()
        # 依目前 text_lines 重新把「ServeCheck 行」對應回 self.serve_checks 的索引
        self._rebuild_servecheck_line_map()

        # --- Landings ---
        # 清空舊的畫面物件
        for art in getattr(self, "_landing_artists", []):
            try: art.remove()
            except Exception: pass
        self._landing_artists = []
        # 預設沒有資料
        self.landings = []

        # 可能是舊 session：沒有 "landings" 就直接略過
        if "landings" in snap:
            self.landings = snap["landings"] or []
            # 依資料重建圖面與 hover meta
            for item in self.landings:
                r_id = item.get("rally_id", 0)
                pred = item.get("pred", [])
                landing = item.get("landing", {})

                yy = [p["pos"]["y"] for p in pred if p.get("pos")]
                zz = [p["pos"]["z"] for p in pred if p.get("pos")]

                if yy and zz:
                    line_pred, = self.ax.plot(yy, zz, '--', linewidth=1.0, alpha=0.6, zorder=3)
                    scat_pred = self.ax.scatter(yy, zz, s=5, alpha=0.6, zorder=4)

                    # 還原 hover meta
                    meta_pred = []
                    for p in pred:
                        pos = p.get("pos", {})
                        meta_pred.append({
                            'x': pos.get('x'), 'y': pos.get('y'), 'z': pos.get('z'),
                            'fid': p.get('id'), 'ts': p.get('timestamp'), 'src': 'pred-traj'
                        })
                    scat_pred.meta = meta_pred

                    # 落地點
                    lpos = (landing or {}).get("pos", {})
                    ly, lz = lpos.get("y"), lpos.get("z")
                    if ly is not None and lz is not None:
                        scat_landing = self.ax.scatter([ly], [lz], s=90, marker='*',
                                                    facecolors='none', edgecolors='lime', linewidths=1.8, zorder=5)
                        scat_landing.meta = [{
                            'x': lpos.get('x'), 'y': ly, 'z': lz,
                            'fid': landing.get('id'), 'ts': landing.get('timestamp'),
                            'src': 'pred-landing'
                        }]
                        for art in (line_pred, scat_pred, scat_landing):
                            art.rally_id = r_id
                            self._landing_artists.append(art)
                    else:
                        # 只有軌跡沒有落地點
                        for art in (line_pred, scat_pred):
                            art.rally_id = r_id
                            self._landing_artists.append(art)
        
        self.balltype_inputs = []
        for item in snap.get("balltype_inputs", []):
            r_id = int(item.get("rally_id", 0))
            pts  = [(float(y), float(z)) for (y, z) in item.get("points", [])]
            meta = []
            for m in item.get("meta", []):
                meta.append({
                    "x": m.get("x"), "y": m.get("y"), "z": m.get("z"),
                    "fid": m.get("fid"), "ts": m.get("ts")
                })
            self.balltype_inputs.append({"rally_id": r_id, "points": pts, "meta": meta})
        
        # 重建 Rally 按鈕
        self.update_rally_buttons()

        # 初次繪製
        self.update_plot()
        self.update_text_display()
        print(f"[Loaded] Session from {path}")

    def _shorten(self, s: str, max_chars: int = 36) -> str:
        if not s or len(s) <= max_chars:
            return s
        return f"{s[:14]}…{s[-18:]}"  # 避免太長擠版面

    def _set_mode_banner(self, mode, name=None):
        if mode == 'live':
            self.mode_text.set_text("Mode: Live")
            self.mode_text.set_color('tab:blue')
        else:
            # View
            suffix = f" - {self._shorten(name)}" if name else ""
            self.mode_text.set_text(f"Mode: View{suffix}")
            self.mode_text.set_color('tab:orange')
        self.fig.canvas.draw_idle()
    
    def _on_mode_changed(self, label: str):
        label = label.lower()
        if label == 'live':
            self.mode = 'live'
            self.ignore_mqtt = False
            self._subscribe_all()
            self.current_loaded_session = None
            self._set_mode_banner('live')
        else:
            self.mode = 'view'
            self.ignore_mqtt = True
            self._unsubscribe_all()
            # 保留上次載入的名稱（若還沒載入就只顯示 Mode: View）
            self._set_mode_banner('view', self.current_loaded_session)

    def _on_load_session_from_ui(self, event=None):
        """讀取 view_name_tb 的文字並載入 session；若留白就用最新的 /drawRecord/<latest>。"""
        name = (self.view_name_tb.text or "").strip()
        if not name:
            # 沒輸入就用 ./drawRecord 內最新一個 session 目錄
            candidate = self._get_latest_record_name()
            if candidate:
                name = candidate
            else:
                print("[Load] No session name provided and no record folder found.")
                return

        path = self._resolve_session_path(name)
        if not path:
            print(f"[Load] Session not found for '{name}'. Expecting /drawRecord/{name}/{name}_session.json")
            return

        # 切換到 View 模式並載入
        self.mode_radio.set_active(1)  # 切到 View
        self._load_session(path)
        self.current_loaded_session = name  # 使用者輸入的不含 .json 的名稱
        self._set_mode_banner('view', name)

        # （可選）清空輸入框
        self.view_name_tb.set_val("")


    def _get_latest_record_name(self):
        """回傳 ./record 底下最新（mtime）的資料夾名稱，沒有則回傳 None。"""
        path = os.path.join(ROOTDIR, "LayerContent", "drawRecord")
        if not os.path.isdir(path):
            return None
        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if not dirs:
            return None
        latest = max(dirs, key=lambda d: os.path.getmtime(os.path.join(path, d)))
        return latest

    def _resolve_session_path(self, name_or_path):
        """
        支援三種輸入：
        1) 只有名稱：       foo            → /drawRecord/foo/foo_session.json
        2) 給資料夾路徑：   /drawRecord/foo   → /drawRecord/foo/foo_session.json 或該夾中唯一 *_session.json
        3) 給 JSON 路徑：   /drawRecord/foo/foo_session.json
        找不到就回傳 None
        """
        p = name_or_path
        # 3) 直接是 json 檔
        if p.endswith(".json") and os.path.isfile(p):
            return p

        # 2) 是資料夾
        if os.path.isdir(p):
            # 先嘗試 name/name_session.json
            base = os.path.basename(os.path.normpath(p))
            cand = os.path.join(p, f"{base}_session.json")
            if os.path.isfile(cand):
                return cand
            # 否則找該夾中唯一的 *_session.json
            jsons = [os.path.join(p, f) for f in os.listdir(p) if f.endswith("_session.json")]
            if len(jsons) == 1:
                return jsons[0]
            return None

        # 1) 就是一個名稱 → ./record/<name>/<name>_session.json
        cand = os.path.join(ROOTDIR, "LayerContent", "drawRecord", name_or_path, f"{name_or_path}_session.json")
        if os.path.isfile(cand):
            return cand
        return None


    # ---- FID 搜尋：字串解析 ----
    def _parse_fid_expr(self, s: str):
        """
        支援：
          - 單一：  "7541"
          - 範圍：  "7000-7050"
          - 多個：  "7001, 7005, 7010-7020"
        回傳 list[('single', v)] 或 ('range', lo, hi)（皆為 int）。
        解析失敗回傳空清單。
        """
        if not s:
            return []
        s = s.strip()
        if not s:
            return []

        # 允許全形逗號
        tokens = [t.strip() for t in s.replace('，', ',').split(',') if t.strip()]
        parts = []
        for tok in tokens:
            if '-' in tok:
                a, b = tok.split('-', 1)
                try:
                    lo, hi = int(a.strip()), int(b.strip())
                    parts.append(('range', min(lo, hi), max(lo, hi)))
                except ValueError:
                    continue
            else:
                try:
                    v = int(tok)
                    parts.append(('single', v))
                except ValueError:
                    continue
        return parts

    def _fid_match(self, fid, parts) -> bool:
        """fid 與解析後的 parts 是否相符。fid 必須能轉成 int。"""
        try:
            v = int(fid)
        except Exception:
            return False

        for p in parts:
            if p[0] == 'single' and v == p[1]:
                return True
            if p[0] == 'range' and p[1] <= v <= p[2]:
                return True
        return False
    
    def _on_fid_submit(self, text: str):
        parts = self._parse_fid_expr(text)
        self._fid_query = parts if parts else None
        self._update_fid_highlight()

    def _clear_fid_highlight(self):
        # 清掉星星與文字
        if self._fid_highlight_art is not None:
            try:
                self._fid_highlight_art.remove()
            except ValueError:
                pass
            self._fid_highlight_art = None
        for t in self._fid_highlight_texts:
            try:
                t.remove()
            except ValueError:
                pass
        self._fid_highlight_texts.clear()

    def _update_fid_highlight(self):
        """
        依 _fid_query 在各來源搜尋 fid 符合的點並高亮：
        - 以星形 marker 大尺寸顯示
        - 上方加上 fid 文字標籤（白底，避免雜訊）
        無查詢則移除高亮。
        """
        # 沒有查詢 → 移除高亮
        if not self._fid_query:
            self._clear_fid_highlight()
            if self.fig:
                self.fig.canvas.draw_idle()
            return

        yy, zz, fids = [], [], []

        # 依序檢查來源：一般點、三類事件、所有 segment 點
        arts = [self.scat_point, self.scat_event_hit, self.scat_event_serve, self.scat_event_dead]
        arts.extend(self.segment_artists)

        for art in arts:
            if art is None:
                continue

            meta = getattr(art, 'meta', [])
            if meta is None:
                meta = []
            offs = art.get_offsets() if hasattr(art, 'get_offsets') else None

            # offs 有可能是單一點/空陣列/None
            if offs is None:
                continue
            try:
                n = min(len(meta), len(offs))
            except TypeError:
                continue
            if n == 0:
                continue

            for i in range(n):
                m = meta[i]
                fid = m.get('fid', None)
                if self._fid_match(fid, self._fid_query):
                    y, z = offs[i]
                    yy.append(y)
                    zz.append(z)
                    fids.append(fid)

        # 先移除舊的高亮
        self._clear_fid_highlight()

        # 畫新的高亮（更醒目）
        if yy:
            self._fid_highlight_art = self.ax.scatter(
                yy, zz,
                s=100, marker='*',                # 大顆星星
                facecolors='yellow', edgecolors='black',
                linewidths=1.2, zorder=6
            )
            # 加上 fid 文字標籤（白底避免被背景干擾）
            # for fid, y, z in zip(fids, yy, zz):
            #     txt = self.ax.text(
            #         y, z + 0.22, str(fid),
            #         fontsize=9, color='black', ha='center', va='bottom',
            #         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            #         zorder=7
            #     )
            #     self._fid_highlight_texts.append(txt)

        if self.fig:
            self.fig.canvas.draw_idle()
    
    
    def clear_data(self, event=None):
        """Clear all stored points, events, and segments, then reset the plot."""
        # 清空 point buffer
        if self.max_points:
            self.point_buffers['point'].queue.clear()
            self.point_buffers['event'].queue.clear()
        else:
            self.point_buffers['point'].clear()
            self.point_buffers['event'].clear()

        # 重新初始化 point buffer
        self.point_buffers = {
            'point': queue.Queue(maxsize=self.max_points) if self.max_points else [],
            'event': queue.Queue(maxsize=self.max_points) if self.max_points else [],
        }

        # 清空 event buffers
        for event_type in self.event_buffers:
            self.event_buffers[event_type].queue.clear()
        
        self.event_buffers = {
            1: queue.Queue(maxsize=self.max_points or 0),
            2: queue.Queue(maxsize=self.max_points or 0),
            3: queue.Queue(maxsize=self.max_points or 0),
        }

        # 清除畫在圖上的 fid 文字
        if hasattr(self, 'event_texts'):
            for text_obj in self.event_texts:
                text_obj.remove()
            self.event_texts.clear()
        else:
            self.event_texts = []

        # 清空 scatter points
        self.scat_event_hit.set_offsets(np.empty((0, 2)))
        self.scat_event_serve.set_offsets(np.empty((0, 2)))
        self.scat_event_dead.set_offsets(np.empty((0, 2)))
        
        # 清除事件序號與 Segment 資料
        self.event_labels.clear()
        self.segment_colors.clear()
        self.segment_points.clear()
        self.segment_data.clear()
        self.segment_counter = 0
        
        # 清除右側文字
        self.text_lines.clear()
        self.update_text_display()
        self.serve_checks = []
        self.line_idx_to_servecheck = {}
        self.selected_servecheck_idx = None
        self._clear_servecheck_highlight()

        # 清除 rally 紀錄跟按鈕
        self.segment_data.clear()
        self.segment_counter = 0
        self.current_rally_id = 0
        self.selected_rally_id = 0
        self.update_rally_buttons()

        # 清除 hover annotation (tooltip)
        if hasattr(self, 'hover_annot'):
            self.hover_annot.set_visible(False)

        for art in self._bridge_artists:
            try:
                art.remove()
            except ValueError:
                pass
        self._bridge_artists.clear()
        
        # 清除 axes 上所有的繪圖物件
        self.ax.clear()
        # 重新設定軸參數
        self.set_axis()
        
        # 重新建立代表 point 與 event 的 scatter 物件
        self.scat_point = self.ax.scatter([], [], s=5, c='grey', alpha=0.3, label='Model3D_point')
        self.scat_event = self.ax.scatter([], [], s=40, c='red', label='Model3D_event')
        self.scat_event_hit = self.ax.scatter([], [], s=40, c='blue', label='Hit Event')
        self.scat_event_serve = self.ax.scatter([], [], s=40, c='red', label='Serve Event')
        self.scat_event_dead = self.ax.scatter([], [], s=40, c='green', label='Dead Event')
        self.scat_balltype = self.ax.scatter([], [], s=5, marker='v', c='orange', label='BallTypeInput', alpha=0.3, zorder=5)
        self.balltype_inputs = []

        # 重新打開 picker
        for art in [self.scat_point, self.scat_event, self.scat_event_hit, self.scat_event_serve, self.scat_event_dead]:
            art.set_picker(5)  # 建議 8~12，比原本 5 好點

        # 同步 overlay 位置（ax 清過之後）
        if hasattr(self, 'ax_overlay') and self.ax_overlay is not None:
            self.ax_overlay.set_position(self.ax.get_position())
        else:
            self.ax_overlay = self.fig.add_axes(self.ax.get_position(), facecolor='none')
            self.ax_overlay.set_axis_off()
        self.ax_overlay.set_zorder(1000)

        # 重新建立 hover annotation 在 overlay 軸上
        self.hover_annot = self.ax_overlay.annotate(
            "", xy=(0,0), xytext=(12, 12),
            xycoords=self.ax.transData,
            textcoords="offset points",
            ha='left', va='bottom',
            bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.95),
            arrowprops=dict(arrowstyle="->", lw=0.5, alpha=0.8),
            zorder=999,
            annotation_clip=False,
        )
        self.hover_annot.set_visible(False)

        
        # 清 Landing 物件
        for art in getattr(self, "_landing_artists", []):
            try: art.remove()
            except Exception: pass
        self._landing_artists = []
        self.landings = []

        # 清 BallTypeInput
        if hasattr(self, 'balltype_inputs'):
            self.balltype_inputs.clear()
        if hasattr(self, 'scat_balltype'):
            self.scat_balltype.set_offsets(np.empty((0,2)))
            self.scat_balltype.meta = []
        
        # 清除 fid 搜尋高亮與輸入
        if hasattr(self, '_fid_highlight_art') or hasattr(self, '_fid_highlight_texts'):
            self._clear_fid_highlight()
        self._fid_query = None
        if hasattr(self, 'fid_tb'):
            try:
                self.fid_tb.set_val("")
            except Exception:
                pass

        self._refresh_balltype_visibility()
        self.fig.canvas.draw_idle()

        print('!!!!! clear')

    def stop(self):
        print("Stopping application...")
        try:
            # output_png = f"./record/{self.timestamp}_fig.png"
            # self.fig.savefig(output_png, dpi=300)
            # print(f"Plot saved as {output_png}")

            self.mqttc.loop_stop()
            self.mqttc.disconnect()
            print("Application stopped.")
        except Exception as e:
            logging.error(f"Error while stopping application: {e}")

    def run(self):
        # Subscribe to relevant topics
        self._subscribe_all()

        # Start plot loop
        # plt.ion()  # Enable interactive mode
        # try:
        #     while True:
        #         if self.new_data_received:
        #             self.update_plot()  # Update the plot
        #             self.new_data_received = False
        #         else:
        #             plt.pause(0.2)  # Pause to avoid freezing
        # except KeyboardInterrupt:
        #     self.stop()
        self._timer = self.fig.canvas.new_timer(interval=50)  # 20 FPS 上限
        self._timer.add_callback(self._on_timer_tick)
        self._timer.start()

        plt.ioff() # 確保關閉互動模式
        plt.show()  # 交給 GUI 自己跑事件回圈（不再用 plt.pause）

    
    def _subscribe_all(self):
        print(">> Subscribing to all topics <<")
        for t in self.SUBS:
            self.mqttc.subscribe(t.format(self.target_devices_name))
            print(f"Subscribed to topic: {t.format(self.target_devices_name)}")

    def _unsubscribe_all(self):
        print(">> Unsubscribing from all topics <<")
        for t in self.SUBS:
            self.mqttc.unsubscribe(t.format(self.target_devices_name))
    
    
    def _on_timer_tick(self):
        # 只在需要時才重繪
        if self.new_data_received:
            self.update_plot()
            self.new_data_received = False

        # 顯示「當前時間/最後收到」那行：降低頻率，且用 draw_idle
        now = time.time()
        if not hasattr(self, "_last_clock_update"):
            self._last_clock_update = 0
        if now - self._last_clock_update >= 1.0:      # 每秒更新一次就好
            self.update_text_display_no_point_message()
            self.fig.canvas.draw_idle()
            self._last_clock_update = now

    def _sync_overlay_position(self, evt=None):
        # 窗口大小變更時，讓 overlay 軸緊貼主圖軸
        if hasattr(self, 'ax_overlay') and self.ax_overlay is not None:
            self.ax_overlay.set_position(self.ax.get_position())
            self.ax_overlay.set_zorder(1000)

    def _on_click_fallback(self, event):
        # 左鍵，且在主圖或 overlay 上才處理
        if event.button != 1 or event.inaxes not in (self.ax, getattr(self, 'ax_overlay', None)):
            return

        # 依優先序檢查的 artists（和 on_hover 一致）
        candidate_arts = []
        candidate_arts.extend(self.segment_artists)
        candidate_arts.extend([self.scat_event_hit, self.scat_event_serve, self.scat_event_dead])
        candidate_arts.append(self.scat_point)
        candidate_arts.append(self.scat_serve_ng)
        if hasattr(self, 'scat_balltype'):
            candidate_arts.append(self.scat_balltype)
        candidate_arts.extend(getattr(self, "_landing_artists", []))

        # 嘗試命中任一可 pick 的 artist
        for art in candidate_arts:
            if not hasattr(art, "contains"):
                continue
            contains, info = art.contains(event)
            if not contains:
                continue

            # 命中 segment scatter → 套用你原本的高亮邏輯
            if hasattr(art, 'segment_id'):
                seg_id = art.segment_id
                if self.selected_segment_id == seg_id:
                    self._clear_segment_highlight()
                    self.selected_segment_id = None
                    self.update_text_display()
                    return
                self.selected_segment_id = seg_id
                self.update_text_display()
                self._draw_segment_highlight(art)
                return
        # 其他類（例如 ServeCheck 小叉叉）暫不處理點選



def signal_handler(signum, frame):
    print("\nReceived signal to terminate.")
    app.stop()
    exit(0)

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Example APP for real-time plotting with MQTT.")
    parser.add_argument("--max-points", type=int, help="Maximum number of points to display (default: all points)")
    parser.add_argument("--office", action="store_true")
    args = parser.parse_args()

    app = ExampleAPP(max_points=args.max_points, office=args.office)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    app.run()

    # signal.sigwait([signal.SIGTERM, signal.SIGINT])
    # app.stop()
