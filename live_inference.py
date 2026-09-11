import logging
import os
import queue
import threading
import time
import tkinter as tk
from collections import deque
from math import sqrt, degrees, acos
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


class PoseDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("天眼测试 V0.39")
        self.root.geometry("1200x800")  # 增加宽度以容纳画廊
        self.root.configure(bg="#0F1419")

        # 设置调试日志
        logging.basicConfig(filename='crimeguard_debug.log', level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # 加载 YOLO 模型
        self.model = YOLO("yolov8n-pose.pt")
        self.conf = 0.7
        self.keypoint_conf = 0.5

        # 用于动作检测的帧历史
        self.prev_keypoints = deque(maxlen=5)
        self.prev_boxes = deque(maxlen=5)
        self.behavior_history = deque(maxlen=3)
        self.punch_counts = {}

        # 视频控制
        self.streaming = False
        self.playing_video = False
        self.paused = False
        self.thread = None
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.current_frame = 0
        self.seek_frame = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30

        # 保存关键帧
        self.frame_queue = queue.Queue(maxsize=100)  # 最多保存 100 帧
        self.saved_frames = []  # 保存帧路径
        self.output_dir = "output_frames"
        os.makedirs(self.output_dir, exist_ok=True)

        # GUI 元素
        self.main_frame = ttk.Frame(self.root, padding=15)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 样式配置
        style = ttk.Style()
        style.configure("Main.TFrame", background="#0F1419")
        style.configure("TButton", font=("Helvetica", 12, "bold"), padding=10, background="#00A3FF",
                        foreground="#FFFFFF")
        style.map("TButton", background=[('active', '#007ACC')])
        style.configure("TLabel", font=("Helvetica", 12), background="#0F1419", foreground="#E6ECF0")
        style.configure("Status.TLabel", font=("Helvetica", 12, "bold"), foreground="#FF4D4D")
        style.configure("Progress.Horizontal.TScale", background="#0F1419", troughcolor="#1F2A44",
                        slidercolor="#00A3FF")
        style.configure("Gallery.TListbox", font=("Helvetica", 10), background="#1F2A44", foreground="#E6ECF0")

        # 标题
        self.label = ttk.Label(
            self.main_frame,
            text="天眼测试 V0.39",
            font=("Helvetica", 22, "bold"),
            foreground="#E6ECF0"
        )
        self.label.grid(row=0, column=0, columnspan=4, pady=(10, 20))

        # 按钮框架
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=1, column=0, columnspan=4, pady=10)

        self.upload_image_btn = ttk.Button(
            self.button_frame, text="上传图片", command=self.upload_image
        )
        self.upload_image_btn.grid(row=0, column=0, padx=8)

        self.upload_video_btn = ttk.Button(
            self.button_frame, text="播放视频", command=self.upload_video
        )
        self.upload_video_btn.grid(row=0, column=1, padx=8)

        self.stream_btn = ttk.Button(
            self.button_frame, text="启动摄像头", command=self.start_stream
        )
        self.stream_btn.grid(row=0, column=2, padx=8)

        self.pause_btn = ttk.Button(
            self.button_frame, text="暂停", command=self.toggle_pause, state="disabled"
        )
        self.pause_btn.grid(row=0, column=3, padx=8)

        self.stop_stream_btn = ttk.Button(
            self.button_frame, text="停止", command=self.stop_stream, state="disabled"
        )
        self.stop_stream_btn.grid(row=0, column=4, padx=8)

        # 图像显示
        self.image_frame = ttk.Frame(self.main_frame, style="Main.TFrame", relief="flat", borderwidth=2)
        self.image_frame.grid(row=2, column=0, columnspan=3, pady=15, sticky="nsew")
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack()

        # 关键帧画廊
        self.gallery_frame = ttk.Frame(self.main_frame, style="Main.TFrame")
        self.gallery_frame.grid(row=2, column=3, padx=10, pady=15, sticky="ns")
        self.gallery_label = ttk.Label(self.gallery_frame, text="关键帧", font=("Helvetica", 12, "bold"),
                                       foreground="#E6ECF0")
        self.gallery_label.pack()
        self.gallery_listbox = tk.Listbox(
            self.gallery_frame, width=30, height=20, bg="#1F2A44", fg="#E6ECF0", font=("Helvetica", 10)
        )
        self.gallery_listbox.pack(pady=5)
        self.gallery_listbox.bind('<<ListboxSelect>>', self.display_selected_frame)

        # 状态标签
        self.status_label = ttk.Label(
            self.main_frame,
            text="状态：就绪",
            style="Status.TLabel",
            wraplength=700
        )
        self.status_label.grid(row=3, column=0, columnspan=4, pady=10)

        # 进度条
        self.progress = ttk.Scale(
            self.main_frame,
            from_=0,
            to=100,
            orient="horizontal",
            style="Progress.Horizontal.TScale",
            command=self.seek_video
        )
        self.progress.grid(row=4, column=0, columnspan=4, sticky="ew", padx=20, pady=5)
        self.progress.configure(state="disabled")

        # 状态栏
        self.status_bar = ttk.Label(
            self.main_frame,
            text="天眼 | 就绪",
            font=("Helvetica", 10),
            relief="sunken",
            anchor="w",
            foreground="#E6ECF0"
        )
        self.status_bar.grid(row=5, column=0, columnspan=4, sticky="ew", pady=10)

        # 启动帧保存线程
        self.frame_save_thread = threading.Thread(target=self.save_frame_worker, daemon=True)
        self.frame_save_thread.start()

    def save_frame_worker(self):
        while True:
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
                frame, filename = frame_data
                cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.saved_frames.append(filename)
                self.root.after(0, lambda: self.gallery_listbox.insert(tk.END, os.path.basename(filename)))
                logging.info(f"保存关键帧: {filename}")
                # 限制保存帧数
                if len(self.saved_frames) > 100:
                    old_frame = self.saved_frames.pop(0)
                    try:
                        os.remove(old_frame)
                        self.root.after(0, lambda: self.gallery_listbox.delete(0))
                        logging.info(f"删除旧帧: {old_frame}")
                    except Exception as e:
                        logging.error(f"删除旧帧失败: {old_frame}, {str(e)}")
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"保存帧错误: {str(e)}")

    def display_selected_frame(self, event):
        selection = self.gallery_listbox.curselection()
        if selection:
            index = selection[0]
            frame_path = self.saved_frames[index]
            try:
                image = Image.open(frame_path)
                image = image.resize((600, 600), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                self.status_bar.configure(text=f"天眼 | 查看关键帧: {os.path.basename(frame_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"加载关键帧失败: {str(e)}")
                logging.error(f"加载关键帧失败: {frame_path}, {str(e)}")

    def calculate_angle(self, p1, p2, p3):
        v1 = [p1[0] - p2[0], p1[1] - p2[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 * mag2 == 0:
            return 0
        cos_angle = dot / (mag1 * mag2)
        cos_angle = min(max(cos_angle, -1), 1)
        return degrees(acos(cos_angle))

    def calculate_distance(self, p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def detect_crime(self, keypoints, keypoint_conf, boxes, frame=None):
        behaviors = []
        debug_info = []
        if len(keypoints) == 0 or len(keypoint_conf) == 0 or len(boxes) == 0:
            debug_info.append("未检测到关键点或边界框")
            logging.debug("未检测到关键点或边界框")
            return behaviors, debug_info

        # 动态阈值
        scale_factor = self.frame_width / 1280 if self.frame_width > 0 else 1
        fps_factor = min(1.0, 30 / self.fps if self.fps > 0 else 1)
        shoulder_dist_threshold = 180 * scale_factor
        collision_dist_threshold = 70 * scale_factor
        wrist_speed_threshold = 60 * scale_factor * fps_factor
        hip_drop_threshold = 100 * (self.frame_height / 720 if self.frame_height > 0 else 1)

        # 处理单人动作
        for i, (person, conf, box) in enumerate(zip(keypoints, keypoint_conf, boxes)):
            if len(person) < 17 or len(conf) < 17:
                debug_info.append(f"人物 {i + 1}: 关键点不完整")
                logging.debug(f"人物 {i + 1}: 关键点不完整")
                continue
            core_keypoints = conf[[5, 6, 7, 8, 9, 10]]
            if len(core_keypoints) == 0 or np.any(core_keypoints < self.keypoint_conf):
                debug_info.append(
                    f"人物 {i + 1}: 核心关键点置信度过低 ({np.min(core_keypoints):.2f} < {self.keypoint_conf:.2f})")
                logging.debug(
                    f"人物 {i + 1}: 核心关键点置信度 ({np.min(core_keypoints):.2f} < {self.keypoint_conf:.2f})")
                continue

            person = person[:, :2].tolist()
            nose = person[0]
            left_shoulder, right_shoulder = person[5], person[6]
            left_elbow, right_elbow = person[7], person[8]
            left_wrist, right_wrist = person[9], person[10]
            left_hip, right_hip = person[11], person[12]
            left_knee, right_knee = person[13], person[14]
            left_ankle, right_ankle = person[15], person[16]

            # 快速挥拳
            if self.prev_keypoints and len(self.prev_keypoints[-1]) > i:
                prev_person = self.prev_keypoints[-1][i]
                if len(prev_person) >= 17:
                    prev_person = prev_person[:, :2].tolist()
                    prev_left_wrist, prev_right_wrist = prev_person[9], prev_person[10]
                    wrist_speed_left = self.calculate_distance(left_wrist, prev_left_wrist)
                    wrist_speed_right = self.calculate_distance(right_wrist, prev_right_wrist)
                    left_elbow_angle = self.calculate_angle(left_wrist, left_elbow, left_shoulder)
                    right_elbow_angle = self.calculate_angle(right_wrist, right_elbow, right_shoulder)
                    if (wrist_speed_left > wrist_speed_threshold or wrist_speed_right > wrist_speed_threshold) and \
                            (left_elbow_angle < 90 or right_elbow_angle < 90):
                        punch_key = f"punch_{i}"
                        self.punch_counts[punch_key] = self.punch_counts.get(punch_key, 0) + 1
                        if self.punch_counts[punch_key] >= 2:
                            behaviors.append(f"快速挥拳 (人物 {i + 1})")
                            logging.debug(
                                f"人物 {i + 1}: 挥拳检测 - 速度: {max(wrist_speed_left, wrist_speed_right):.2f}, 角度: {min(left_elbow_angle, right_elbow_angle):.2f}, 次数: {self.punch_counts[punch_key]}")
                    elif (wrist_speed_left <= wrist_speed_threshold and wrist_speed_right <= wrist_speed_threshold):
                        debug_info.append(
                            f"人物 {i + 1}: 腕部速度不足 ({max(wrist_speed_left, wrist_speed_right):.2f} < {wrist_speed_threshold:.2f})")
                        logging.debug(
                            f"人物 {i + 1}: 腕部速度不足 ({max(wrist_speed_left, wrist_speed_right):.2f} < {wrist_speed_threshold:.2f})")

            # 其他动作
            body_angle = self.calculate_angle(nose, [(left_shoulder[0] + right_shoulder[0]) / 2,
                                                     (left_shoulder[1] + right_shoulder[1]) / 2],
                                              [left_hip[0], left_hip[1]])
            if nose[1] > min(left_shoulder[1], right_shoulder[1]) and \
                    (left_wrist[1] > left_hip[1] or right_wrist[1] > right_hip[1]) and body_angle > 30:
                behaviors.append(f"偷窃动作 (人物 {i + 1})")

            if self.prev_keypoints and len(self.prev_keypoints[-1]) > i:
                prev_person = self.prev_keypoints[-1][i]
                if len(prev_person) >= 17:
                    prev_person = prev_person[:, :2].tolist()
                    prev_ankle = prev_person[15]
                    stride = self.calculate_distance(left_ankle, right_ankle)
                    speed = self.calculate_distance(left_ankle, prev_ankle)
                    if stride > 220 * scale_factor and speed > 90 * scale_factor * fps_factor:
                        behaviors.append(f"逃逸跑动 (人物 {i + 1})")

            if (left_hip[1] > left_knee[1] - 35 * (self.frame_height / 720)) and \
                    (right_hip[1] > right_knee[1] - 35 * (self.frame_height / 720)):
                behaviors.append(f"隐秘蹲伏 (人物 {i + 1})")

            if self.prev_keypoints and len(self.prev_keypoints[-1]) > i:
                prev_person = self.prev_keypoints[-1][i]
                if len(prev_person) >= 17:
                    prev_person = prev_person[:, :2].tolist()
                    prev_hip = [(prev_person[11][0] + prev_person[12][0]) / 2,
                                (prev_person[11][1] + prev_person[12][1]) / 2]
                    curr_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
                    hip_drop = curr_hip[1] - prev_hip[1]
                    if hip_drop > hip_drop_threshold:
                        behaviors.append(f"摔倒动作 (人物 {i + 1})")

            if self.prev_keypoints and len(self.prev_keypoints[-1]) > i:
                prev_person = self.prev_keypoints[-1][i]
                if len(prev_person) >= 17:
                    prev_person = prev_person[:, :2].tolist()
                    prev_left_wrist, prev_right_wrist = prev_person[9], prev_person[10]
                    wrist_speed_left = self.calculate_distance(left_wrist, prev_left_wrist)
                    wrist_speed_right = self.calculate_distance(right_wrist, prev_right_wrist)
                    if (left_wrist[1] < left_shoulder[1] or right_wrist[1] < right_shoulder[1]) and \
                            (wrist_speed_left > wrist_speed_threshold or wrist_speed_right > wrist_speed_threshold) and \
                            (left_wrist[1] > prev_left_wrist[1] or right_wrist[1] > prev_right_wrist[1]):
                        behaviors.append(f"破坏动作 (人物 {i + 1})")

        # 多人交互动作
        if len(keypoints) >= 2:
            for i in range(len(keypoints)):
                for j in range(i + 1, len(keypoints)):
                    person1, person2 = keypoints[i][:, :2].tolist(), keypoints[j][:, :2].tolist()
                    conf1, conf2 = keypoint_conf[i], keypoint_conf[j]
                    box1, box2 = boxes[i], boxes[j]
                    core_keypoints1 = conf1[[5, 6, 7, 8, 9, 10]]
                    core_keypoints2 = conf2[[5, 6, 7, 8, 9, 10]]
                    if len(core_keypoints1) == 0 or len(core_keypoints2) == 0 or \
                            np.any(core_keypoints1 < self.keypoint_conf) or np.any(
                        core_keypoints2 < self.keypoint_conf):
                        debug_info.append(
                            f"人物 {i + 1} ↔ {j + 1}: 核心关键点置信度过低 ({min(np.min(core_keypoints1), np.min(core_keypoints2)):.2f} < {self.keypoint_conf:.2f})")
                        logging.debug(
                            f"人物 {i + 1} ↔ {j + 1}: 核心关键点置信度 ({min(np.min(core_keypoints1), np.min(core_keypoints2)):.2f} < {self.keypoint_conf:.2f})")
                        continue

                    p1_shoulder = [(person1[5][0] + person1[6][0]) / 2, (person1[5][1] + person1[6][1]) / 2]
                    p2_shoulder = [(person2[5][0] + person2[6][0]) / 2, (person2[5][1] + person2[6][1]) / 2]
                    p1_hip = [(person1[11][0] + person1[12][0]) / 2, (person1[11][1] + person1[12][1]) / 2]
                    p2_hip = [(person2[11][0] + person2[12][0]) / 2, (person2[11][1] + person2[12][1]) / 2]
                    p1_wrist, p2_wrist = person1[9], person2[9]
                    p1_elbow, p2_elbow = person1[7], person2[7]

                    # 斗殴动作
                    shoulder_dist = self.calculate_distance(p1_shoulder, p2_shoulder)
                    if shoulder_dist < shoulder_dist_threshold:
                        fight_components = []
                        torso1 = [p1_shoulder, p1_hip]
                        torso2 = [p2_shoulder, p2_hip]
                        torso_dist = min(
                            self.calculate_distance(torso1[0], torso2[0]),
                            self.calculate_distance(torso1[0], torso2[1]),
                            self.calculate_distance(torso1[1], torso2[0]),
                            self.calculate_distance(torso1[1], torso2[1])
                        )
                        if torso_dist < collision_dist_threshold:
                            fight_components.append("身体冲撞")
                            logging.debug(
                                f"人物 {i + 1} ↔ {j + 1}: 身体碰撞 (距离: {torso_dist:.2f} < {collision_dist_threshold:.2f})")

                        arm1 = [p1_wrist, p1_elbow]
                        arm2 = [p2_wrist, p2_elbow]
                        arm_dist = min(
                            self.calculate_distance(arm1[0], arm2[0]),
                            self.calculate_distance(arm1[0], arm2[1]),
                            self.calculate_distance(arm1[1], arm2[0]),
                            self.calculate_distance(arm1[1], arm2[1])
                        )
                        if arm_dist < collision_dist_threshold:
                            fight_components.append("手臂交叉")
                            logging.debug(
                                f"人物 {i + 1} ↔ {j + 1}: 手臂交叉 (距离: {arm_dist:.2f} < {collision_dist_threshold:.2f})")

                        punch_detected = False
                        if self.prev_keypoints and len(self.prev_keypoints[-1]) > i and len(
                                self.prev_keypoints[-1]) > j:
                            prev_p1_wrist = self.prev_keypoints[-1][i][9][:2].tolist()
                            prev_p2_wrist = self.prev_keypoints[-1][j][9][:2].tolist()
                            wrist_speed1 = self.calculate_distance(p1_wrist, prev_p1_wrist)
                            wrist_speed2 = self.calculate_distance(p2_wrist, prev_p2_wrist)
                            wrist_angle1 = self.calculate_angle(p1_wrist, person1[7], person1[5])
                            wrist_angle2 = self.calculate_angle(p2_wrist, person2[7], person2[5])
                            if (wrist_speed1 > wrist_speed_threshold or wrist_speed2 > wrist_speed_threshold) and \
                                    (wrist_angle1 < 90 or wrist_angle2 < 90):
                                punch_key1 = f"punch_{i}"
                                punch_key2 = f"punch_{j}"
                                self.punch_counts[punch_key1] = self.punch_counts.get(punch_key1, 0) + 1
                                self.punch_counts[punch_key2] = self.punch_counts.get(punch_key2, 0) + 1
                                if self.punch_counts[punch_key1] >= 2 or self.punch_counts[punch_key2] >= 2:
                                    fight_components.append("快速挥拳")
                                    punch_detected = True
                                    logging.debug(
                                        f"人物 {i + 1} ↔ {j + 1}: 挥拳 (速度: {max(wrist_speed1, wrist_speed2):.2f}, 角度: {min(wrist_angle1, wrist_angle2):.2f}, 次数: {max(self.punch_counts[punch_key1], self.punch_counts[punch_key2])})")
                            if not punch_detected:
                                debug_info.append(
                                    f"人物 {i + 1} ↔ {j + 1}: 腕部速度不足 ({max(wrist_speed1, wrist_speed2):.2f} < {wrist_speed_threshold:.2f})")
                                logging.debug(
                                    f"人物 {i + 1} ↔ {j + 1}: 腕部速度不足 ({max(wrist_speed1, wrist_speed2):.2f} < {wrist_speed_threshold:.2f})")

                        if len(fight_components) >= 2:
                            behaviors.append(f"斗殴动作 (人物 {i + 1} ↔ {j + 1}: {', '.join(fight_components)})")
                            logging.debug(f"斗殴动作检测: 人物 {i + 1} ↔ {j + 1}, 特征: {', '.join(fight_components)}")

                    # 推搡动作
                    if shoulder_dist < shoulder_dist_threshold:
                        if self.prev_keypoints and len(self.prev_keypoints[-1]) > i:
                            prev_p1_wrist = self.prev_keypoints[-1][i][9][:2].tolist()
                            wrist_speed = self.calculate_distance(p1_wrist, prev_p1_wrist)
                            wrist_angle = self.calculate_angle(p1_wrist, person1[7], person1[5])
                            if wrist_speed > 100 * scale_factor * fps_factor and wrist_angle > 120:
                                behaviors.append(f"推搡动作 (人物 {i + 1} -> {j + 1})")

                    # 尾随动作
                    nose_to_shoulder_dist = self.calculate_distance(person1[0], p2_shoulder)
                    if nose_to_shoulder_dist < 150 * scale_factor and person1[0][1] < p2_shoulder[1]:
                        if self.prev_keypoints and len(self.prev_keypoints[-1]) > i and len(
                                self.prev_keypoints[-1]) > j:
                            prev_p1_nose = self.prev_keypoints[-1][i][0][:2].tolist()
                            prev_p2_shoulder = [
                                (self.prev_keypoints[-1][j][5][0] + self.prev_keypoints[-1][j][6][0]) / 2,
                                (self.prev_keypoints[-1][j][5][1] + self.prev_keypoints[-1][j][6][1]) / 2]
                            speed_diff = self.calculate_distance(prev_p1_nose, person1[0]) - self.calculate_distance(
                                p2_shoulder, prev_p2_shoulder)
                            if abs(speed_diff) < 20 * scale_factor * fps_factor:
                                behaviors.append(f"尾随动作 (人物 {i + 1} -> {j + 1})")

                    # 抢夺动作
                    wrist_to_shoulder_dist = self.calculate_distance(p1_wrist, p2_shoulder)
                    if wrist_to_shoulder_dist < 50 * scale_factor:
                        if self.prev_keypoints and len(self.prev_keypoints[-1]) > i:
                            prev_p1_wrist = self.prev_keypoints[-1][i][9][:2].tolist()
                            wrist_speed = self.calculate_distance(p1_wrist, prev_p1_wrist)
                            if wrist_speed > 100 * scale_factor * fps_factor:
                                behaviors.append(f"抢夺动作 (人物 {i + 1} -> {j + 1})")

        # 保存关键帧
        if behaviors and frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            behavior_str = "_".join(behaviors).replace(" ", "_").replace(":", "_")
            frame_filename = os.path.join(self.output_dir, f"frame_{self.current_frame}_{timestamp}_{behavior_str}.jpg")
            try:
                self.frame_queue.put((frame, frame_filename))
                self.status_bar.configure(
                    text=f"天眼 | 保存关键帧: {os.path.basename(frame_filename)} | 已保存 {len(self.saved_frames) + 1} 帧")
            except queue.Full:
                logging.warning("关键帧队列已满，跳过保存")
                debug_info.append("关键帧队列已满")

        # 时间窗口
        self.behavior_history.append(set(behaviors))
        if len(self.behavior_history) >= 3:
            common_behaviors = set.intersection(*list(self.behavior_history))
            fight_behaviors = [b for b in common_behaviors if "斗殴动作" in b]
            if fight_behaviors:
                return fight_behaviors, debug_info
            if not common_behaviors and not behaviors:
                debug_info.append("未检测到持续动作")
                logging.debug("未检测到持续动作")
            return list(common_behaviors), debug_info
        debug_info.append("行为历史不足 3 帧")
        logging.debug("行为历史不足 3 帧")
        return [], debug_info

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                results = self.model(source=file_path, conf=self.conf)
                image_array = results[0].plot(conf=True, boxes=True)
                keypoints = results[0].keypoints.xy.cpu().numpy()
                keypoint_conf = results[0].keypoints.conf.cpu().numpy() if results[
                    0].keypoints.has_visible else np.zeros_like(keypoints)
                boxes = results[0].boxes.xyxy.cpu().numpy()

                behaviors, debug_info = self.detect_crime(keypoints, keypoint_conf, boxes, image_array)
                status_text = "状态：" + (
                    ", ".join(behaviors) if behaviors else f"未检测到可疑行为 ({'; '.join(debug_info)})")
                self.status_label.configure(text=status_text)
                self.status_bar.configure(text=f"天眼 | 处理图片：{os.path.basename(file_path)}")

                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_array)
                image = image.resize((600, 600), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
            except Exception as e:
                messagebox.showerror("错误", f"处理图片失败：{str(e)}")
                self.status_label.configure(text="状态：处理图片失败")
                logging.error(f"处理图片失败：{str(e)}")

    def upload_video(self):
        self.stop_stream()
        file_path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
        if file_path:
            try:
                if not file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                    messagebox.showwarning("警告", "建议使用 .mp4 或 .avi 格式以确保兼容性")

                self.video_path = file_path
                self.playing_video = True
                self.paused = False
                self.seek_frame = None
                self.stream_btn.configure(state="disabled")
                self.stop_stream_btn.configure(state="normal")
                self.pause_btn.configure(state="normal", text="暂停")
                self.progress.configure(state="normal")
                self.cap = cv2.VideoCapture(self.video_path)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                if self.fps < 15:
                    messagebox.showwarning("警告", f"视频帧率 {self.fps:.1f} FPS 较低，可能影响动作检测")
                logging.info(f"视频信息：分辨率 {self.frame_width}x{self.frame_height}, 帧率 {self.fps:.1f}")
                self.cap.release()
                self.thread = threading.Thread(target=self.play_video)
                self.thread.daemon = True
                self.thread.start()
                self.status_bar.configure(text=f"天眼 | 播放视频：{os.path.basename(file_path)} | 0%")
            except Exception as e:
                messagebox.showerror("错误", f"启动视频播放失败：{str(e)}")
                self.status_label.configure(text="状态：启动视频失败")
                logging.error(f"启动视频播放失败：{str(e)}")

    def toggle_pause(self):
        if self.playing_video:
            self.paused = not self.paused
            self.pause_btn.configure(text="继续" if self.paused else "暂停")
            progress = (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0
            self.status_bar.configure(
                text=f"天眼 | {'暂停' if self.paused else '播放'}视频：{os.path.basename(self.video_path)} | {progress:.1f}% | 已保存 {len(self.saved_frames)} 帧")

    def seek_video(self, value):
        if self.playing_video and self.total_frames > 0:
            self.seek_frame = int(float(value) * self.total_frames / 100)

    def play_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("错误", "无法打开视频"))
            self.stop_stream()
            logging.error("无法打开视频")
            return

        self.current_frame = 0
        frame_skip = 1 if self.fps >= 15 else 2
        frame_count = 0
        while self.playing_video and self.cap.isOpened():
            if self.paused:
                time.sleep(0.1)
                continue
            try:
                if self.seek_frame is not None:
                    self.current_frame = self.seek_frame
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    self.seek_frame = None

                success, frame = self.cap.read()
                if not success:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                results = self.model(frame, conf=self.conf)
                annotated_frame = results[0].plot(conf=True, boxes=True)
                keypoints = results[0].keypoints.xy.cpu().numpy()
                keypoint_conf = results[0].keypoints.conf.cpu().numpy() if results[
                    0].keypoints.has_visible else np.zeros_like(keypoints)
                boxes = results[0].boxes.xyxy.cpu().numpy()

                behaviors, debug_info = self.detect_crime(keypoints, keypoint_conf, boxes, annotated_frame)
                status_text = "状态：" + (
                    ", ".join(behaviors) if behaviors else f"未检测到可疑行为 ({'; '.join(debug_info)})")
                self.root.after(0, lambda: self.status_label.configure(text=status_text))

                if len(keypoints) > 0:
                    self.prev_keypoints.append(keypoints)
                    self.prev_boxes.append(boxes)

                self.current_frame += 1
                progress = (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0
                self.root.after(0, lambda: self.progress.set(progress))
                self.root.after(0, lambda: self.status_bar.configure(
                    text=f"天眼 | 播放视频：{os.path.basename(self.video_path)} | {progress:.1f}% | 已保存 {len(self.saved_frames)} 帧"))

                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(annotated_frame)
                img = img.resize((600, 600), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.root.after(0, lambda: self.image_label.configure(image=photo))
                self.image_label.image = photo

                time.sleep(1.0 / (self.fps if self.fps >= 15 else 15))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"视频播放错误：{str(e)}"))
                logging.error(f"视频播放错误：{str(e)}")
                break

        if self.cap:
            self.cap.release()
        self.root.after(0, self.stop_stream)

    def start_stream(self):
        if not self.streaming and not self.playing_video:
            self.streaming = True
            self.stream_btn.configure(state="disabled")
            self.stop_stream_btn.configure(state="normal")
            self.thread = threading.Thread(target=self.update_stream)
            self.thread.daemon = True
            self.thread.start()

    def update_stream(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("错误", "无法打开摄像头"))
            self.stop_stream()
            logging.error("无法打开摄像头")
            return

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        logging.info(f"摄像头信息：分辨率 {self.frame_width}x{self.frame_height}, 帧率 {self.fps:.1f}")
        self.current_frame = 0
        while self.streaming and self.cap.isOpened():
            try:
                success, frame = self.cap.read()
                if not success:
                    break

                results = self.model(frame, conf=self.conf)
                annotated_frame = results[0].plot(conf=True, boxes=True)
                keypoints = results[0].keypoints.xy.cpu().numpy()
                keypoint_conf = results[0].keypoints.conf.cpu().numpy() if results[
                    0].keypoints.has_visible else np.zeros_like(keypoints)
                boxes = results[0].boxes.xyxy.cpu().numpy()

                behaviors, debug_info = self.detect_crime(keypoints, keypoint_conf, boxes, annotated_frame)
                status_text = "状态：" + (
                    ", ".join(behaviors) if behaviors else f"未检测到可疑行为 ({'; '.join(debug_info)})")
                self.root.after(0, lambda: self.status_label.configure(text=status_text))
                self.root.after(0, lambda: self.status_bar.configure(
                    text=f"天眼 | 摄像头流 | 已保存 {len(self.saved_frames)} 帧"))

                if len(keypoints) > 0:
                    self.prev_keypoints.append(keypoints)
                    self.prev_boxes.append(boxes)

                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(annotated_frame)
                img = img.resize((600, 600), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.root.after(0, lambda: self.image_label.configure(image=photo))
                self.image_label.image = photo

                time.sleep(0.03)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"流错误：{str(e)}"))
                logging.error(f"流错误：{str(e)}")
                break

        if self.cap:
            self.cap.release()
        self.root.after(0, self.stop_stream)

    def stop_stream(self):
        self.streaming = False
        self.playing_video = False
        self.paused = False
        self.seek_frame = None
        self.stream_btn.configure(state="normal")
        self.stop_stream_btn.configure(state="disabled")
        self.pause_btn.configure(state="disabled", text="暂停")
        self.progress.configure(state="disabled")
        self.image_label.configure(image="")
        self.status_label.configure(text="状态：就绪")
        self.status_bar.configure(text=f"测试 v3.9 | 就绪 | 已保存 {len(self.saved_frames)} 帧")
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.thread:
            self.thread = None
        self.video_path = None
        self.prev_keypoints.clear()
        self.prev_boxes.clear()
        self.behavior_history.clear()
        self.punch_counts.clear()
        self.current_frame = 0
        self.total_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30


if __name__ == "__main__":
    root = tk.Tk()
    app = PoseDetectionApp(root)
    root.mainloop()