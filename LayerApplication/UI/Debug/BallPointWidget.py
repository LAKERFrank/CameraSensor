import pyqtgraph as pg
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QHBoxLayout, 
                             QLabel, QSpinBox, QTextEdit, QTableWidget, 
                             QTableWidgetItem, QCheckBox, QHeaderView)
from PyQt5.QtCore import Qt, QTimer
from lib.point import Point
import numpy as np
from collections import deque
import json
import random

class BallPointWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()
        self.points_data = deque(maxlen=1000)  # 限制最大點數，避免記憶體溢出
        self.pending_updates = deque()  # 待更新的點
        self.all_y_coords = []  # 儲存所有 Y 座標
        self.all_z_coords = []  # 儲存所有 Z 座標
        
        # Segment 相關資料
        self.segment_colors = []  # 儲存段的顏色
        self.segment_points = []  # 儲存段的點
        self.segment_counter = 0  # 段計數器
        self.segment_data = []  # 儲存段資料
        
        # Event 相關資料
        self.event_buffers = {
            1: deque(maxlen=100),  # Hit 事件
            2: deque(maxlen=100),  # Serve 事件
            3: deque(maxlen=100),  # Dead 事件
        }
        
        # Rally 相關資料
        self.rally_data = []  # 儲存 rally 資料
        self.current_rally = None  # 當前正在進行的 rally
        self.rally_counter = 0  # rally 計數器
        self.rally_visibility = {}  # 儲存每個 rally 的可見性狀態
        
        # 文字顯示相關
        self.text_lines = []  # 儲存文字行
        self.follow_latest = True  # 是否跟隨最新
        self.show_speed_labels = True  # 是否在圖中顯示速度標籤
        
        # 事件描述
        self.EVENT_DESCRIPTIONS = {
            1: "Hit",           # 擊球
            2: "Serve",         # 發球
            3: "Dead",          # 死球
            4: "Non-FreeFly",   # 不自由飛行
            5: "NoBall/Static"  # 無球/靜止
        }
        
        self.setupUpdateTimer()
    
    def setupUI(self):
        layout = QHBoxLayout()  # 改為水平佈局
        
        # 左側：圖表區域
        left_layout = QVBoxLayout()
        
        # 創建 PyQtGraph 的 GraphicsLayoutWidget
        self.graph_widget = pg.GraphicsLayoutWidget()
        
        # 創建 PlotItem
        self.plot_item = self.graph_widget.addPlot()
        
        # 設置座標軸範圍
        self.plot_item.setXRange(-7, 7)
        self.plot_item.setYRange(0, 6)
        
        # 設置標籤
        self.plot_item.setLabel('left', 'Z Coordinate')
        self.plot_item.setLabel('bottom', 'Y Coordinate')
        
        # 添加網格
        self.plot_item.showGrid(x=True, y=True)
        
        # 效能優化：設置繪圖選項
        self.plot_item.setDownsampling(auto=True, mode='peak')
        
        # 創建散點圖項目，使用更高效的設置
        self.scatter_item = pg.ScatterPlotItem(
            pen=None, 
            symbol='o', 
            size=8, 
            brush='gray',
            pxMode=True  # 像素模式，更高效
        )
        self.plot_item.addItem(self.scatter_item)
        
        # 創建事件散點圖項目
        self.scatter_event_hit = pg.ScatterPlotItem(
            pen=None, symbol='o', size=12, brush='blue', pxMode=True
        )
        self.scatter_event_serve = pg.ScatterPlotItem(
            pen=None, symbol='o', size=12, brush='red', pxMode=True
        )
        self.scatter_event_dead = pg.ScatterPlotItem(
            pen=None, symbol='o', size=12, brush='green', pxMode=True
        )
        
        self.plot_item.addItem(self.scatter_event_hit)
        self.plot_item.addItem(self.scatter_event_serve)
        self.plot_item.addItem(self.scatter_event_dead)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        # 清除按鈕
        self.btn_clear_ball = QPushButton('清除球點')
        self.btn_clear_ball.clicked.connect(self.clearBallPoints)
        
        # 清除事件按鈕
        self.btn_clear_events = QPushButton('清除事件')
        self.btn_clear_events.clicked.connect(self.clearEvents)
        
        # 清除段按鈕
        self.btn_clear_segments = QPushButton('清除段')
        self.btn_clear_segments.clicked.connect(self.clearSegments)
        
        # 點數顯示
        self.point_count_label = QLabel('點數: 0')
        
        # 速度標籤顯示控制
        self.speed_label_checkbox = QCheckBox('顯示速度標籤')
        self.speed_label_checkbox.setChecked(True)
        self.speed_label_checkbox.stateChanged.connect(self.toggleSpeedLabels)
        
        control_layout.addWidget(self.btn_clear_ball)
        control_layout.addWidget(self.btn_clear_events)
        control_layout.addWidget(self.btn_clear_segments)
        control_layout.addWidget(self.point_count_label)
        control_layout.addWidget(self.speed_label_checkbox)
        control_layout.addStretch()
        
        left_layout.addWidget(self.graph_widget)
        left_layout.addLayout(control_layout)
        
        # 右側：文字顯示區域
        right_layout = QVBoxLayout()
        
        # 狀態標籤
        self.status_label = QLabel('狀態: 準備就緒')
        right_layout.addWidget(self.status_label)
        
        # 文字顯示區域
        self.text_display = QTextEdit()
        self.text_display.setMaximumWidth(300)
        self.text_display.setReadOnly(True)
        self.text_display.setMaximumHeight(200)  # 恢復原來的高度
        right_layout.addWidget(self.text_display)
        
        # Rally 表格標籤
        rally_label = QLabel('Rally 列表')
        rally_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        right_layout.addWidget(rally_label)
        
        # Rally 表格
        self.rally_table = QTableWidget()
        self.rally_table.setMaximumWidth(300)
        self.rally_table.setMaximumHeight(150)
        self.rally_table.setColumnCount(4)
        self.rally_table.setHorizontalHeaderLabels(['顯示', 'Rally ID', '開始', '結束'])
        
        # 設置表格屬性
        header = self.rally_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)  # 顯示欄位固定寬度
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Rally ID 欄位自動調整
        header.setSectionResizeMode(2, QHeaderView.Stretch)  # 開始欄位自動調整
        header.setSectionResizeMode(3, QHeaderView.Stretch)  # 結束欄位自動調整
        
        self.rally_table.setColumnWidth(0, 50)  # 顯示欄位寬度
        right_layout.addWidget(self.rally_table)
        
        # 右側控制按鈕
        right_control_layout = QHBoxLayout()
        
        self.btn_clear_text = QPushButton('清除文字')
        self.btn_clear_text.clicked.connect(self.clearText)
        
        self.btn_follow_latest = QPushButton('跟隨最新')
        self.btn_follow_latest.setCheckable(True)
        self.btn_follow_latest.setChecked(True)
        self.btn_follow_latest.clicked.connect(self.toggleFollowLatest)
        
        self.btn_clear_rally = QPushButton('清除 Rally')
        self.btn_clear_rally.clicked.connect(self.clearRally)
        
        right_control_layout.addWidget(self.btn_clear_text)
        right_control_layout.addWidget(self.btn_follow_latest)
        right_control_layout.addWidget(self.btn_clear_rally)
        right_layout.addLayout(right_control_layout)
        
        right_layout.addStretch()
        
        # 組合左右佈局
        layout.addLayout(left_layout, 2)  # 左側佔 2/3
        layout.addLayout(right_layout, 1)  # 右側佔 1/3
        
        self.setLayout(layout)
        
        # 效能設置 - 固定關閉效能模式
        self.performance_mode = False
        self.batch_update = False  # 即時更新模式
        self.show_trajectory = False  # 固定關閉軌跡
    
    def setupUpdateTimer(self):
        """設置定時器進行批次更新"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.processPendingUpdates)
        self.update_timer.start(33)  # 30 FPS 更新頻率
    
    def processPendingUpdates(self):
        """處理待更新的點"""
        if not self.pending_updates:
            return
            
        # 批次處理所有待更新的點
        while self.pending_updates:
            points = self.pending_updates.popleft()
            self.addNewPoints(points)
        
        # 更新顯示
        self.updateDisplay()
    
    def addNewPoints(self, points):
        """添加新的點到現有數據中"""
        if not points:
            return
            
        for point in points:
            self.all_y_coords.append(point.y)
            self.all_z_coords.append(point.z)
            self.points_data.append((point.y, point.z, point.fid))  # 添加 fid 追蹤
    
    def updateDisplay(self):
        """更新顯示"""
        # 檢查是否有已結束的 Rally
        has_finished_rally = len(self.rally_data) > 0
        
        # 球點顯示
        if self.all_y_coords and self.all_z_coords:
            if has_finished_rally:
                # 如果有已結束的 Rally，隱藏對應時間範圍內的球點
                self.hideRallyTimeRangePoints()
            else:
                # 如果沒有已結束的 Rally，顯示所有球點
                self.scatter_item.setData(self.all_y_coords, self.all_z_coords)
            self.point_count_label.setText(f'點數: {len(self.points_data)}')

        # 事件點顯示
        if not has_finished_rally:
            self.updateEventPoints()
        else:
            # 隱藏對應時間範圍內的事件點
            self.hideRallyTimeRangeEvents()

        # 段顯示
        if not has_finished_rally:
            self.updateSegmentDisplay()
        else:
            # 隱藏對應時間範圍內的段和速度標籤
            self.hideRallyTimeRangeSegments()
        
        # Rally 顯示
        self.updateRallyDisplay()
        # 文字顯示
        self.updateTextDisplay()
    
    def updateEventPoints(self):
        """更新事件點顯示"""
        # 清除現有事件點
        self.scatter_event_hit.setData([], [])
        self.scatter_event_serve.setData([], [])
        self.scatter_event_dead.setData([], [])
        
        # 更新各類型事件點
        event_category_data = {
            1: {'scatter': self.scatter_event_hit, 'color': 'blue'},
            2: {'scatter': self.scatter_event_serve, 'color': 'red'},
            3: {'scatter': self.scatter_event_dead, 'color': 'green'},
        }
        
        for event_type, buffer in self.event_buffers.items():
            if buffer:
                y_data = [point[0] for point in buffer]
                z_data = [point[1] for point in buffer]
                event_category_data[event_type]['scatter'].setData(y_data, z_data)
    
    def updateSegmentDisplay(self):
        """更新段顯示"""
        # 清除現有段和速度標籤
        for item in self.plot_item.items[:]:
            if hasattr(item, '_is_segment') and item._is_segment:
                self.plot_item.removeItem(item)
            if hasattr(item, '_is_speed_label') and item._is_speed_label:
                self.plot_item.removeItem(item)
        
        # 重新繪製所有段和速度標籤
        for i, (segment_color, segment_points) in enumerate(zip(self.segment_colors, self.segment_points)):
            if segment_points:
                y_data, z_data = zip(*segment_points)
                segment_scatter = pg.ScatterPlotItem(
                    pen=None, symbol='o', size=6, 
                    brush='blue', pxMode=True  # 改為藍色
                )
                segment_scatter._is_segment = True  # 標記為段項目
                segment_scatter.setData(y_data, z_data)
                self.plot_item.addItem(segment_scatter)
                
                # 添加速度標籤
                if (self.show_speed_labels and 
                    i < len(self.segment_data) and 
                    self.segment_data[i].get("speed")):
                    speed_info = self.segment_data[i]["speed"]
                    if speed_info and speed_info.get("combined_speed"):
                        speed_value = speed_info["combined_speed"]
                        
                        # 在段的第一個點位置顯示速度標籤
                        if segment_points:
                            first_y, first_z = segment_points[0]
                            
                            # 計算第幾球標示
                            rally_id, ball_number = self.getRallyBallNumber(self.segment_data[i].get("fids", [None])[0] if self.segment_data[i].get("fids") else None)
                            if rally_id and ball_number:
                                label_text = f'{ball_number}: {speed_value:.1f} km/hr'
                            else:
                                label_text = f'{speed_value:.1f} km/hr'
                            
                            speed_text = pg.TextItem(
                                text=label_text,
                                color=(255, 255, 255),
                                border=pg.mkPen(color=(0, 0, 0), width=1),
                                fill=pg.mkBrush(color=(0, 0, 0, 180))
                            )
                            speed_text._is_speed_label = True
                            speed_text.setPos(first_y, first_z + 0.3)
                            self.plot_item.addItem(speed_text)
    
    def updateTextDisplay(self):
        """更新文字顯示"""
        try:
            max_follow_lines = 35
            total_lines = len(self.text_lines)
            
            if self.follow_latest:
                visible_lines = self.text_lines[-max_follow_lines:]
                start = len(self.text_lines) - len(visible_lines)
            else:
                start = 0
                visible_lines = self.text_lines[:max_follow_lines]
            
            segment_lines = sum(1 for line in visible_lines if "Segment" in line)
            event_lines = sum(1 for line in visible_lines if "Event" in line)
            
            status_text = f"總計: {total_lines}, 段: {segment_lines}, 事件: {event_lines}"
            self.status_label.setText(status_text)
            
            # 更新文字顯示區域
            display_text = ""
            for i, line in enumerate(visible_lines):
                line_number = start + i + 1
                display_text += f"{line_number:>4}: {line}\n"
            
            self.text_display.setText(display_text)
            
        except Exception as e:
            print(f"更新文字顯示時發生錯誤: {e}")
    
    def updateSpeedDisplay(self):
        """更新速度資訊顯示"""
        # (刪除 updateSpeedDisplay 方法的全部內容)
    
    def updateBallPoints(self, points):
        """
        更新球點顯示（保留現有點，即時更新）
        Args:
            points: Point 物件列表
        """
        if not points:
            return
        
        # 即時更新模式：直接添加點並更新顯示
        self.addNewPoints(points)
        self.updateDisplay()
    
    def addBallPoints(self, points):
        """
        添加新的球點（不覆蓋現有的）
        Args:
            points: Point 物件列表
        """
        if not points:
            return
            
        self.addNewPoints(points)
        
        # 將球點添加到當前 Rally
        if self.current_rally:
            for point in points:
                ball_point_data = {
                    "fid": point.fid,
                    "position": [point.x, point.y, point.z]
                }
                self.current_rally["ball_points"].append(ball_point_data)
        
        self.updateDisplay()
    
    def processSegmentMessage(self, segment_data):
        """
        處理段訊息
        Args:
            segment_data: 段資料字典，包含軌跡點資訊和速度資訊
        """
        try:
            print('segment_data:', segment_data)
            # 增加段計數器
            self.segment_counter += 1
            segment_id = self.segment_counter
            
            # 生成隨機顏色
            segment_color = (random.random(), random.random(), random.random())
            
            # 提取點座標和速度資訊
            segment_points = []
            segment_fids = []
            speed_info = None
            ball_type = None

            # 新格式：包含速度資訊的 segment
            start_fid = segment_data.get("start_fid", "N/A")
            end_fid = segment_data.get("end_fid", "N/A")
            speed = segment_data.get("speed", [0, 0, 0])
            ball_type = segment_data.get("ball_type", "N/A")
            
            # 計算合併速度（km/hr）
            combined_speed = (speed[0]**2 + speed[1]**2 + speed[2]**2) ** 0.5  # m/s
            combined_speed *= 3.6  # 轉換為 km/hr
            
            speed_info = {
                "speed_vector": speed,
                "combined_speed": combined_speed
            }
            
            # 使用 start_position 作為點
            start_pos = segment_data.get('start_position', [0, 0, 0])
            y = start_pos[1]
            z = start_pos[2]
            segment_points.append((y, z))
            segment_fids.append(start_fid)
            
            # 添加文字記錄（包含速度資訊）
            new_line = f"Segment: {start_fid} - {end_fid}, Speed: {combined_speed:.2f} km/hr, Ball Type: {ball_type}"
            self.text_lines.append(new_line)
            
            # 儲存段資料
            self.segment_data.append({
                "segment_id": segment_id,
                "points": segment_points,
                "fids": segment_fids,
                "speed": speed_info,
                "ball_type": ball_type
            })
            
            # 儲存顯示資料
            self.segment_colors.append(segment_color)
            self.segment_points.append(segment_points)
            
            # 更新顯示
            self.updateDisplay()
            
        except Exception as e:
            print(f"處理段訊息時發生錯誤: {e}")
    
    def processEventMessage(self, event_data):
        """
        處理事件訊息
        Args:
            event_data: 事件資料字典
        """
        try:
            fid = event_data.get("fid", "N/A")
            event_type = event_data.get("event", None)
            position = event_data.get("position")
            
            if event_type in self.event_buffers and position:
                y, z = position[1], position[2]
                data = (y, z, fid)
                buffer = self.event_buffers[event_type]
                buffer.append(data)
            
            # 處理 Rally 邏輯
            self.processRallyLogic(event_type, fid, position)
            
            # 添加文字記錄
            event_desc = self.EVENT_DESCRIPTIONS.get(event_type, f"Unknown({event_type})")
            new_line = f"Event: {fid}, Type: {event_desc}"
            self.text_lines.append(new_line)
            
            # 更新顯示
            self.updateDisplay()
            
        except Exception as e:
            print(f"處理事件訊息時發生錯誤: {e}")
    
    def processRallyLogic(self, event_type, fid, position):
        """
        處理 Rally 邏輯
        Args:
            event_type: 事件類型
            fid: 幀 ID
            position: 位置資訊
        """
        try:
            if event_type == 2:  # Serve 事件 - 開始新的 rally
                self.startNewRally(fid, position)
            elif event_type == 3:  # Dead 事件 - 結束當前 rally
                self.endCurrentRally(fid, position)
            elif event_type == 1 and self.current_rally:  # Hit 事件 - 添加到當前 rally
                self.addHitToRally(fid, position)
                
        except Exception as e:
            print(f"處理 Rally 邏輯時發生錯誤: {e}")
    
    def startNewRally(self, fid, position):
        """
        開始新的 rally
        Args:
            fid: 幀 ID
            position: 位置資訊
        """
        # 如果當前有未完成的 rally，先結束它
        if self.current_rally:
            self.endCurrentRally(fid, position)
        
        # 創建新的 rally
        self.rally_counter += 1
        self.current_rally = {
            "rally_id": self.rally_counter,
            "start_fid": fid,
            "start_position": position,
            "end_fid": None,
            "end_position": None,
            "hits": [],
            "ball_points": [],
            "color": (random.random(), random.random(), random.random())
        }
        
        # 預設隱藏 Rally
        self.rally_visibility[self.rally_counter] = False
        
        print(f"開始新的 Rally {self.rally_counter} (FID: {fid})")
    
    def endCurrentRally(self, fid, position):
        """
        結束當前 rally
        Args:
            fid: 幀 ID
            position: 位置資訊
        """
        if self.current_rally:
            self.current_rally["end_fid"] = fid
            self.current_rally["end_position"] = position
            self.rally_data.append(self.current_rally.copy())
            print(f"結束 Rally {self.current_rally['rally_id']} (FID: {fid})")
            # 預設隱藏 Rally
            self.rally_visibility[self.current_rally["rally_id"]] = False
            self.current_rally = None
            self.updateRallyTable()
    
    def addHitToRally(self, fid, position):
        """
        添加擊球到當前 rally
        Args:
            fid: 幀 ID
            position: 位置資訊
        """
        if self.current_rally:
            hit_data = {
                "fid": fid,
                "position": position
            }
            self.current_rally["hits"].append(hit_data)
    
    def updateRallyTable(self):
        """更新 Rally 表格"""
        try:
            self.rally_table.setRowCount(len(self.rally_data))
            
            for row, rally in enumerate(self.rally_data):
                # 顯示 checkbox
                checkbox = QCheckBox()
                checkbox.setChecked(self.rally_visibility.get(rally["rally_id"], True))
                checkbox.stateChanged.connect(lambda state, rid=rally["rally_id"]: self.toggleRallyVisibility(rid, state))
                self.rally_table.setCellWidget(row, 0, checkbox)
                
                # Rally ID
                self.rally_table.setItem(row, 1, QTableWidgetItem(f"Rally {rally['rally_id']}"))
                
                # 開始幀
                start_fid = rally.get("start_fid", "N/A")
                self.rally_table.setItem(row, 2, QTableWidgetItem(str(start_fid)))
                
                # 結束幀
                end_fid = rally.get("end_fid", "進行中")
                self.rally_table.setItem(row, 3, QTableWidgetItem(str(end_fid)))
                
        except Exception as e:
            print(f"更新 Rally 表格時發生錯誤: {e}")
    
    def toggleRallyVisibility(self, rally_id, state):
        """
        切換 Rally 可見性
        Args:
            rally_id: Rally ID
            state: checkbox 狀態
        """
        self.rally_visibility[rally_id] = (state == Qt.Checked)
        self.updateRallyDisplay()
        
        # 如果有已結束的 Rally，需要重新更新段和速度標籤的顯示
        if len(self.rally_data) > 0:
            self.hideRallyTimeRangeSegments()
    
    def updateRallyDisplay(self):
        """更新 Rally 顯示"""
        # 清除現有 rally 項目
        items_to_remove = []
        for item in self.plot_item.items:
            if hasattr(item, '_is_rally') and item._is_rally:
                items_to_remove.append(item)
        
        for item in items_to_remove:
            self.plot_item.removeItem(item)
        
        # 重新繪製可見的 rally
        for rally in self.rally_data:
            if self.rally_visibility.get(rally["rally_id"], True):
                self.drawRally(rally)
    
    def drawRally(self, rally):
        """
        繪製單個 rally，依據點類型分色
        Args:
            rally: rally 資料
        """
        try:
            # 收集 rally 的所有點與型別
            rally_points_with_type = []
            # Serve
            if rally.get("start_position"):
                pos = rally["start_position"]
                fid = rally.get("start_fid", 0)
                rally_points_with_type.append((fid, pos[1], pos[2], 'serve'))
            # Ball points
            for ball_point in rally.get("ball_points", []):
                pos = ball_point["position"]
                fid = ball_point.get("fid", 0)
                rally_points_with_type.append((fid, pos[1], pos[2], 'ball'))
            # Hit
            for hit in rally.get("hits", []):
                pos = hit["position"]
                fid = hit.get("fid", 0)
                rally_points_with_type.append((fid, pos[1], pos[2], 'hit'))
            # Dead
            if rally.get("end_position"):
                pos = rally["end_position"]
                fid = rally.get("end_fid", 0)
                rally_points_with_type.append((fid, pos[1], pos[2], 'dead'))
            
            # 分類
            serve_pts = [(y, z) for _, y, z, t in rally_points_with_type if t == 'serve']
            hit_pts = [(y, z) for _, y, z, t in rally_points_with_type if t == 'hit']
            dead_pts = [(y, z) for _, y, z, t in rally_points_with_type if t == 'dead']
            ball_pts = [(y, z) for _, y, z, t in rally_points_with_type if t == 'ball']
            
            # 繪製順序：先畫球點（底層），再畫事件點（上層）
            
            # 1. 先畫球點（灰色，較小）
            if ball_pts:
                ball_scatter = pg.ScatterPlotItem(pen=None, symbol='o', size=8, brush='gray', pxMode=True)
                ball_scatter._is_rally = True
                y, z = zip(*ball_pts)
                ball_scatter.setData(y, z)
                self.plot_item.addItem(ball_scatter)
            
            # 2. 再畫事件點（彩色，較大，在上層）
            if serve_pts:
                serve_scatter = pg.ScatterPlotItem(pen=None, symbol='o', size=12, brush='red', pxMode=True)
                serve_scatter._is_rally = True
                y, z = zip(*serve_pts)
                serve_scatter.setData(y, z)
                self.plot_item.addItem(serve_scatter)
            
            if hit_pts:
                hit_scatter = pg.ScatterPlotItem(pen=None, symbol='o', size=12, brush='blue', pxMode=True)
                hit_scatter._is_rally = True
                y, z = zip(*hit_pts)
                hit_scatter.setData(y, z)
                self.plot_item.addItem(hit_scatter)
            
            if dead_pts:
                dead_scatter = pg.ScatterPlotItem(pen=None, symbol='o', size=12, brush='green', pxMode=True)
                dead_scatter._is_rally = True
                y, z = zip(*dead_pts)
                dead_scatter.setData(y, z)
                self.plot_item.addItem(dead_scatter)
                
        except Exception as e:
            print(f"繪製 Rally 時發生錯誤: {e}")
    
    def clearBallPoints(self):
        """清除所有球點"""
        self.scatter_item.setData([], [])
        self.points_data.clear()
        self.pending_updates.clear()
        self.all_y_coords.clear()
        self.all_z_coords.clear()
        self.point_count_label.setText('點數: 0')
    
    def clearSegments(self):
        """清除所有段"""
        self.segment_colors.clear()
        self.segment_points.clear()
        self.segment_data.clear()
        self.segment_counter = 0
        
        # 清除圖表中的段項目和速度標籤
        items_to_remove = []
        for item in self.plot_item.items:
            if hasattr(item, '_is_segment') and item._is_segment:
                items_to_remove.append(item)
            if hasattr(item, '_is_speed_label') and item._is_speed_label:
                items_to_remove.append(item)
        
        for item in items_to_remove:
            self.plot_item.removeItem(item)
        
        # 重置速度顯示
        self.updateSpeedDisplay()
    
    def clearEvents(self):
        """清除所有事件"""
        for buffer in self.event_buffers.values():
            buffer.clear()
        
        # 清除事件散點圖
        self.scatter_event_hit.setData([], [])
        self.scatter_event_serve.setData([], [])
        self.scatter_event_dead.setData([], [])
    
    def clearText(self):
        """清除文字顯示"""
        self.text_lines.clear()
        self.text_display.clear()
        self.status_label.setText('狀態: 已清除文字')
    
    def toggleFollowLatest(self):
        """切換是否跟隨最新"""
        self.follow_latest = self.btn_follow_latest.isChecked()
        self.updateTextDisplay()
    
    def reset(self):
        """重置 widget 狀態"""
        self.clearBallPoints()
        self.clearSegments()
        self.clearEvents()
        self.clearRally()
        self.clearText()
    
    def getPointCount(self):
        """獲取當前顯示的點數"""
        return len(self.points_data)
    
    def getSegmentCount(self):
        """獲取當前段數量"""
        return len(self.segment_data)
    
    def getEventCount(self):
        """獲取當前事件數量"""
        total_events = 0
        for buffer in self.event_buffers.values():
            total_events += len(buffer)
        return total_events
    
    def setMaxPoints(self, max_points):
        """
        設置最大點數限制
        Args:
            max_points: 最大點數
        """
        self.points_data = deque(self.points_data, maxlen=max_points)
        # 同步更新座標列表
        if len(self.points_data) < len(self.all_y_coords):
            self.all_y_coords = [p[0] for p in self.points_data]
            self.all_z_coords = [p[1] for p in self.points_data]
    
    def setUpdateFPS(self, fps):
        """
        設置更新頻率
        Args:
            fps: 每秒更新次數
        """
        interval = max(1, int(1000 / fps))
        self.update_timer.setInterval(interval)
    
    def enableDownsampling(self, enable=True):
        """
        啟用/禁用下採樣（效能優化）
        Args:
            enable: 是否啟用下採樣
        """
        if enable:
            self.plot_item.setDownsampling(auto=True, mode='peak')
        else:
            self.plot_item.setDownsampling(auto=False)
    
    def getTrajectoryData(self):
        """
        獲取軌跡數據
        Returns:
            tuple: (y_coords, z_coords)
        """
        return self.all_y_coords.copy(), self.all_z_coords.copy()
    
    def getSegmentData(self):
        """
        獲取段數據
        Returns:
            list: 段資料列表
        """
        return self.segment_data.copy()
    
    def getEventData(self):
        """
        獲取事件數據
        Returns:
            dict: 事件緩衝區資料
        """
        return {k: list(v) for k, v in self.event_buffers.items()}
    
    def getRallyData(self):
        """
        獲取 Rally 數據
        Returns:
            list: Rally 資料列表
        """
        return self.rally_data.copy()
    
    def getRallyCount(self):
        """
        獲取當前 Rally 數量
        Returns:
            int: Rally 數量
        """
        return len(self.rally_data)
    
    def getCurrentRally(self):
        """
        獲取當前正在進行的 Rally
        Returns:
            dict: 當前 Rally 資料，如果沒有則返回 None
        """
        return self.current_rally
    
    def setPointColor(self, color):
        """
        設置點顏色
        Args:
            color: 顏色字符串或 QColor 物件
        """
        self.scatter_item.setBrush(color)
    
    def clearRally(self):
        """清除所有 Rally"""
        self.rally_data.clear()
        self.current_rally = None
        self.rally_counter = 0
        self.rally_visibility.clear()
        
        # 清除圖表中的 rally 項目
        items_to_remove = []
        for item in self.plot_item.items:
            if hasattr(item, '_is_rally') and item._is_rally:
                items_to_remove.append(item)
        
        for item in items_to_remove:
            self.plot_item.removeItem(item)
        
        # 清空表格
        self.rally_table.setRowCount(0)
        
        print("已清除所有 Rally")
    
    def hideRallyTimeRangePoints(self):
        """隱藏 Rally 時間範圍內的球點"""
        try:
            # 收集所有 Rally 的時間範圍
            rally_ranges = []
            for rally in self.rally_data:
                start_fid = rally.get("start_fid", 0)
                end_fid = rally.get("end_fid", 0)
                if start_fid and end_fid:
                    rally_ranges.append((start_fid, end_fid))
            
            # 過濾出不在 Rally 時間範圍內的球點
            visible_y_coords = []
            visible_z_coords = []
            
            for point_data in self.points_data:
                y, z, fid = point_data  # 現在 points_data 包含 (y, z, fid)
                
                # 檢查是否在任何 Rally 時間範圍內
                in_rally_range = False
                for start_fid, end_fid in rally_ranges:
                    if start_fid <= fid <= end_fid:
                        in_rally_range = True
                        break
                
                # 如果不在 Rally 範圍內，則顯示
                if not in_rally_range:
                    visible_y_coords.append(y)
                    visible_z_coords.append(z)
            
            # 更新顯示
            self.scatter_item.setData(visible_y_coords, visible_z_coords)
            
        except Exception as e:
            print(f"隱藏 Rally 時間範圍內球點時發生錯誤: {e}")
    
    def hideRallyTimeRangeEvents(self):
        """隱藏 Rally 時間範圍內的事件點"""
        try:
            # 收集所有 Rally 的時間範圍
            rally_ranges = []
            for rally in self.rally_data:
                start_fid = rally.get("start_fid", 0)
                end_fid = rally.get("end_fid", 0)
                if start_fid and end_fid:
                    rally_ranges.append((start_fid, end_fid))
            
            # 過濾各類型事件點
            event_category_data = {
                1: {'scatter': self.scatter_event_hit, 'visible_data': []},
                2: {'scatter': self.scatter_event_serve, 'visible_data': []},
                3: {'scatter': self.scatter_event_dead, 'visible_data': []},
            }
            
            for event_type, buffer in self.event_buffers.items():
                if buffer:
                    visible_data = []
                    for point in buffer:
                        y, z, fid = point
                        
                        # 檢查是否在任何 Rally 時間範圍內
                        in_rally_range = False
                        for start_fid, end_fid in rally_ranges:
                            if start_fid <= fid <= end_fid:
                                in_rally_range = True
                                break
                        
                        # 如果不在 Rally 範圍內，則顯示
                        if not in_rally_range:
                            visible_data.append((y, z))
                    
                    # 更新顯示
                    if visible_data:
                        y_data, z_data = zip(*visible_data)
                        event_category_data[event_type]['scatter'].setData(y_data, z_data)
                    else:
                        event_category_data[event_type]['scatter'].setData([], [])
                else:
                    event_category_data[event_type]['scatter'].setData([], [])
                    
        except Exception as e:
            print(f"隱藏 Rally 時間範圍內事件點時發生錯誤: {e}")
    
    def hideRallyTimeRangeSegments(self):
        """隱藏 Rally 時間範圍內的段和速度標籤"""
        try:
            # 收集所有可見 Rally 的時間範圍
            visible_rally_ranges = []
            for rally in self.rally_data:
                if self.rally_visibility.get(rally["rally_id"], True):
                    start_fid = rally.get("start_fid", 0)
                    end_fid = rally.get("end_fid", 0)
                    if start_fid and end_fid:
                        visible_rally_ranges.append((start_fid, end_fid))
            
            # 清除現有段和速度標籤
            for item in self.plot_item.items[:]:
                if hasattr(item, '_is_segment') and item._is_segment:
                    self.plot_item.removeItem(item)
                if hasattr(item, '_is_speed_label') and item._is_speed_label:
                    self.plot_item.removeItem(item)
            
            # 重新繪製段和速度標籤
            for i, (segment_color, segment_points) in enumerate(zip(self.segment_colors, self.segment_points)):
                if segment_points and i < len(self.segment_data):
                    segment_data = self.segment_data[i]
                    segment_fid = segment_data.get("fids", [None])[0] if segment_data.get("fids") else None
                    
                    # 檢查是否在任何可見 Rally 時間範圍內
                    in_visible_rally_range = False
                    if segment_fid is not None:
                        for start_fid, end_fid in visible_rally_ranges:
                            if start_fid <= segment_fid <= end_fid:
                                in_visible_rally_range = True
                                break
                    
                    # 檢查是否在任何已結束 Rally 的時間範圍內
                    in_any_rally_range = False
                    if segment_fid is not None:
                        for rally in self.rally_data:
                            start_fid = rally.get("start_fid", 0)
                            end_fid = rally.get("end_fid", 0)
                            if start_fid and end_fid and start_fid <= segment_fid <= end_fid:
                                in_any_rally_range = True
                                break
                    
                    # 決定是否顯示：
                    # 1. 如果段不在任何已結束 Rally 範圍內（即時資料），則顯示
                    # 2. 如果段在已結束 Rally 範圍內，則根據 Rally 可見性決定
                    should_show = False
                    if not in_any_rally_range:
                        # 即時資料，始終顯示
                        should_show = True
                    elif in_visible_rally_range:
                        # 在可見 Rally 範圍內，顯示
                        should_show = True
                    
                    if should_show:
                        y_data, z_data = zip(*segment_points)
                        segment_scatter = pg.ScatterPlotItem(
                            pen=None, symbol='o', size=6, 
                            brush='blue', pxMode=True
                        )
                        segment_scatter._is_segment = True
                        segment_scatter.setData(y_data, z_data)
                        self.plot_item.addItem(segment_scatter)
                        
                        # 添加速度標籤
                        if (self.show_speed_labels and segment_data.get("speed")):
                            speed_info = segment_data["speed"]
                            if speed_info and speed_info.get("combined_speed"):
                                speed_value = speed_info["combined_speed"]
                                
                                if segment_points:
                                    first_y, first_z = segment_points[0]
                                    
                                    # 計算第幾球標示
                                    rally_id, ball_number = self.getRallyBallNumber(segment_fid)
                                    if rally_id and ball_number:
                                        label_text = f'{ball_number}: {speed_value:.1f} km/hr'
                                    else:
                                        label_text = f'{speed_value:.1f} km/hr'
                                    
                                    speed_text = pg.TextItem(
                                        text=label_text,
                                        color=(255, 255, 255),
                                        border=pg.mkPen(color=(0, 0, 0), width=1),
                                        fill=pg.mkBrush(color=(0, 0, 0, 180))
                                    )
                                    speed_text._is_speed_label = True
                                    speed_text.setPos(first_y, first_z + 0.3)
                                    self.plot_item.addItem(speed_text)
                    
        except Exception as e:
            print(f"隱藏 Rally 時間範圍內段和速度標籤時發生錯誤: {e}")
    
    def toggleSpeedLabels(self, state):
        """
        切換速度標籤顯示
        Args:
            state: checkbox 狀態
        """
        self.show_speed_labels = (state == Qt.Checked)
        self.updateSegmentDisplay()
    
    def setSpeedLabelsVisible(self, visible):
        """
        設置速度標籤是否可見
        Args:
            visible: 是否顯示速度標籤
        """
        self.show_speed_labels = visible
        self.speed_label_checkbox.setChecked(visible)
        self.updateSegmentDisplay()
    
    def isSpeedLabelsVisible(self):
        """
        獲取速度標籤是否可見
        Returns:
            bool: 是否顯示速度標籤
        """
        return self.show_speed_labels

    def getRallyBallNumber(self, segment_fid):
        """
        計算段屬於哪個 Rally 的第幾次擊球
        Args:
            segment_fid: 段的幀 ID
        Returns:
            tuple: (rally_id, ball_number) 或 (None, None) 如果不屬於任何 Rally
        """
        if segment_fid is None:
            return None, None
            
        for rally in self.rally_data:
            start_fid = rally.get("start_fid", 0)
            end_fid = rally.get("end_fid", 0)
            if start_fid and end_fid and start_fid <= segment_fid <= end_fid:
                # 計算這是第幾次擊球
                # 發球是第1次擊球
                if segment_fid == start_fid:
                    ball_number = 1
                else:
                    # 計算在 hits 中是第幾次擊球
                    hits = rally.get("hits", [])
                    ball_number = 1  # 發球算第1次
                    for hit in hits:
                        ball_number += 1
                        if segment_fid == hit.get("fid", 0):
                            break
                
                return rally["rally_id"], ball_number
        
        return None, None
