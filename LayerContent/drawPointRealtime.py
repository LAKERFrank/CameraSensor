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

from datetime import datetime
from matplotlib.widgets import Button

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
            self.mqttc.connect("140.113.213.131", 1884, 60)
        else:
            try:
                ipaddress.ip_address(broker_ip)
                self.mqttc.connect(broker_ip, broker_port)
            except:
                logging.error("Fail to connect MQTT Broker.")

        self.mqttc.loop_start()

        self.new_data_received = False

        # Initialize plot with two subplots: one for plotting and one for text display on the right side
        self.fig, (self.ax, self.ax_text) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1.5]})
        self.office = office
        self.set_axis()

        self.text_lines = []  # Store text lines for segment information

        # Scatter plots for different topics:
        # "point" for normal points (使用透明灰色)
        self.scat_point = self.ax.scatter([], [], s=5, c='grey', alpha=0.5, label='Model3D_point')
        # "event" for event points (使用紅色)
        self.scat_event = self.ax.scatter([], [], s=40, c='red', label='Model3D_event')
        self.event_labels = []  # 用於存放 event 序號
        self.scat_event_hit = self.ax.scatter([], [], s=40, c='blue', label='Event 1 (Hit)')
        self.scat_event_serve = self.ax.scatter([], [], s=40, c='red', label='Event 2 (Serve)')
        self.scat_event_dead = self.ax.scatter([], [], s=40, c='green', label='Event 3 (Dead)')

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
        self.segment_data = []  # Save segment data (id, points, fid)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        # ↓ Latest button
        ax_go_bottom = self.fig.add_axes([0.9, 0.05, 0.05, 0.03])
        self.btn_go_bottom = Button(ax_go_bottom, '↓ Latest')
        self.btn_go_bottom.on_clicked(self.on_go_bottom_clicked)

        self.text_offset = 0
        self.text_display_count = 35

        self.follow_latest = True

        # Clear button
        self.ax_button = self.fig.add_axes([0.8, 0.95, 0.075, 0.04])
        self.button = Button(self.ax_button, 'Clear')
        self.button.on_clicked(self.clear_data)
      
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

    def on_connect(self, client: mqtt.Client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code}")

    def on_message(self, client, userdata, msg):
        self.new_data_received = True
        # Process based on topic
        if msg.topic == f"/DATA/{self.target_devices_name}/LayerContent/Model3D/Point":
            self.process_point_message(msg)
            # print("=== point ===")
        elif msg.topic == f"/DATA/{self.target_devices_name}/LayerContent/Model3D/Event/Debug":
            print(f"[ApplicationLayer] Received on topic '{msg.topic}'")
            self.process_event_message_debug(msg)
        elif msg.topic == f"/DATA/{self.target_devices_name}/LayerContent/Model3D/Segment/Debug":
            print(f"[ApplicationLayer] Received on topic '{msg.topic}'")
            self.process_segment_message_debug(msg)
        
        elif msg.topic == f"/DATA/{self.target_devices_name}/LayerContent/Model3D/Event":
            print()
            print(f"Event")
            print(datetime.now().strftime("%H:%M:%S"))
            print(msg.payload)
            self.process_event_message(msg)

        elif msg.topic == f"/DATA/{self.target_devices_name}/LayerContent/Model3D/Segment":
            print()
            print(f"Segment")
            print(datetime.now().strftime("%H:%M:%S"))
            print(msg.payload)
            self.process_segment_message(msg)

    
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
                self.update_text_display_no_point_message()
            # else:
            #     self.update_text_display()  # Update with regular info
                
    
    def update_text_display_no_point_message(self):
        """Display the current time when no point message is received for 3 minutes."""
        # self.ax_text.cla()  # Clear previous text
        # self.ax_text.axis('off')
        current_time = datetime.now().strftime("%H:%M:%S")
        last_received_formatted = datetime.fromtimestamp(self.last_received_time).strftime("%H:%M:%S")
        
        # self.ax_text.text(0.01, 1 - 0.03, f"Last received time: {current_time}", fontsize=10, color='orange', ha='left', va='top')
        # self.fig.canvas.draw_idle()

        self.last_received_text.set_text(f"Current Time: {current_time}, Last received: {last_received_formatted}")
        # self.fig.canvas.draw_idle()
        # self.ax_last_received.figure.canvas.draw_idle()

    
    
    def process_point_message_org(self, msg, category):
        """Process individual point messages for Model3D (normal points) and Event."""
        if category == 'point':
            self.last_received_time = time.time()
        
        payload = json.loads(msg.payload)
        for point in payload.get('linear', []):
            y = point['pos']['y']
            z = point['pos']['z']
            fid = point.get('id', 'NoID')

            # Add new point to the corresponding buffer
            if self.max_points:
                if self.point_buffers[category].full():
                    self.point_buffers[category].get()  # Remove the oldest point
                self.point_buffers[category].put((y, z))
            else:
                self.point_buffers[category].append((y, z))  # Keep all points when no limit

            # 如果是 event 類型，則記錄序號
            if category == 'event':
                seq = len(self.event_labels) + 1  # Event 序號
                self.event_labels.append((seq, fid))
                print(self.event_labels, point.get('id', 'NoID'))

    def process_point_message(self, msg):
        """Process individual point messages for Model3D points."""
        self.last_received_time = time.time()
        # print(self.last_receiver_time)
        # print(time.time())
        # print(datetime.fromtimestamp(self.last_received_time).strftime("%H:%M:%S"))

        payload = json.loads(msg.payload)
        for point in payload.get('linear', []):
            y = point['pos']['y']
            z = point['pos']['z']
            fid = point.get('id', 'NoID')

            # Add new point to the buffer
            if self.max_points:
                if self.point_buffers['point'].full():
                    self.point_buffers['point'].get()  # Remove the oldest point
                self.point_buffers['point'].put((y, z))
            else:
                self.point_buffers['point'].append((y, z))  # Keep all points when no limit

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
        segment_id = self.segment_counter

        # Generate a random color for the new segment
        segment_color = (random.random(), random.random(), random.random())

        # Extract points and fid from the segment
        segment_points = []
        segment_fids = []
        for point in trajectory:
            y = point['pos']['y']
            z = point['pos']['z']
            fid = point.get('id', 'N/A')
            segment_points.append((y, z))
            segment_fids.append(fid)

        # Save segment data for logging or exporting
        self.segment_data.append({
            "segment_id": segment_id,
            "points": segment_points,
            "fids": segment_fids
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

        if event_type in self.event_buffers and position:
            y, z = position[1], position[2]
            data = (y, z, fid)
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

        new_line = f"Segment : {start_fid} - {end_fid}, Speed: {combined_speed:.2f}, Ball Type: {ball_type}"

        # Append the new line to text_lines and update text display
        self.text_lines.append(new_line)
        self.update_text_display()

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
            
            for i, line in enumerate(visible_lines):
                line_number = start + i + 1
                line = f"{line_number:>4}:    {line}"

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

                self.ax_text.text(
                    0.01, 1 - i * 0.03, line, fontsize=10, color=color, ha='left', va='top'
                )

            self.fig.canvas.draw_idle()

        except Exception as e:
            logging.error(f"Error updating text display: {e}")

    def update_plot(self):
        # Update normal points ("point")
        point_points = (
            list(self.point_buffers['point'].queue) if self.max_points else self.point_buffers['point']
        )
        if point_points:
            y_data, z_data = zip(*point_points)
            self.scat_point.set_offsets(list(zip(y_data, z_data)))

        # Update event points by type
        self.scat_event_hit.set_offsets(np.empty((0, 2)))
        self.scat_event_serve.set_offsets(np.empty((0, 2)))
        self.scat_event_dead.set_offsets(np.empty((0, 2)))
        event_category_data = {
            1: {'scatter': self.scat_event_hit, 'color': 'blue'},
            2: {'scatter': self.scat_event_serve, 'color': 'red'},
            3: {'scatter': self.scat_event_dead, 'color': 'green'},
        }

        if self.office:
            y_offset = 0.2
        else:
            y_offset = 0.5
            
        for event_type, buffer in self.event_buffers.items():
            if isinstance(buffer, queue.Queue):
                data = list(buffer.queue)
            else:
                data = buffer

            if data:
                y_data, z_data = zip(*[(y, z) for y, z, _ in data])
                event_category_data[event_type]['scatter'].set_offsets(list(zip(y_data, z_data)))

                for y, z, fid in data:
                    self.ax.text(
                        (y - y_offset) if y < 0 else y, z - 0.25, f"{fid}",
                        fontsize=9, color=event_category_data[event_type]['color'],
                        ha='left', va='bottom'
                    )


        # Update trajectory segments
        for segment_color, segment_points in zip(self.segment_colors, self.segment_points):
            y_data, z_data = zip(*segment_points)
            self.ax.scatter(y_data, z_data, s=5, c=[segment_color])

        if self.fig is None:
            print("Warning: Figure is not initialized!")
            return
        
        try:
            self.fig.canvas.draw()  # Update the plot
            # self.fig.canvas.flush_events()
        except Exception as e:
            logging.error(f"Error during draw: {e}. Resetting ticks.")
            self.ax.set_xticks(range(self.x_min, self.x_max+1))
            self.ax.set_yticks(range(self.y_min, self.y_max+1))

    
    def on_key_press(self, event):
        if event.key == "c":  # 按下 c 鍵
            self.clear_data()


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

        # 清除 axes 上所有的繪圖物件
        self.ax.clear()
        # 重新設定軸參數
        self.set_axis()
        
        # 重新建立代表 point 與 event 的 scatter 物件
        self.scat_point = self.ax.scatter([], [], s=5, c='grey', alpha=0.5, label='Model3D_point')
        self.scat_event = self.ax.scatter([], [], s=40, c='red', label='Model3D_event')
        self.scat_point = self.ax.scatter([], [], s=5, c='grey', alpha=0.5, label='Model3D_point')
        self.scat_event_hit = self.ax.scatter([], [], s=40, c='blue', label='Hit Event')
        self.scat_event_serve = self.ax.scatter([], [], s=40, c='red', label='Serve Event')
        self.scat_event_dead = self.ax.scatter([], [], s=40, c='green', label='Dead Event')


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
        self.mqttc.subscribe(f"/DATA/{self.target_devices_name}/LayerContent/Model3D/Point")
        self.mqttc.subscribe(f"/DATA/{self.target_devices_name}/LayerContent/Model3D/Event/Debug")
        self.mqttc.subscribe(f"/DATA/{self.target_devices_name}/LayerContent/Model3D/Segment/Debug")

        self.mqttc.subscribe(f"/DATA/{self.target_devices_name}/LayerContent/Model3D/Event")
        self.mqttc.subscribe(f"/DATA/{self.target_devices_name}/LayerContent/Model3D/Segment")

        # Start plot loop
        plt.ion()  # Enable interactive mode
        try:
            while True:
                if self.new_data_received:
                    self.update_plot()  # Update the plot
                    self.new_data_received = False
                else:
                    plt.pause(0.2)  # Pause to avoid freezing
        except KeyboardInterrupt:
            self.stop()

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
