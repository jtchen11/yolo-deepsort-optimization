# gui.py - 基于Tkinter的图形界面（完整修复版）
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import os
import threading
import time
from datetime import datetime
import webbrowser

# 导入项目核心模块
from self_utils.multi_tasks import Counting_Processing
from self_utils.overall_method import Object_Counter, Image_Capture
from deep_sort.configs.parser import get_config
from deep_sort.deep_sort import DeepSort
from self_utils.globals_val import Global
import imutils
import torch.nn as nn
from packaging import version


class VideoTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多目标检测与跟踪系统")
        self.root.geometry("1000x830")
        self.root.resizable(True, True)

        # ========== 状态变量 ==========
        self.is_playing = False
        self.is_paused = False
        self.is_camera_mode = False
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.videowriter = None
        self.total_frame_count = 0
        self.fps = 25
        self.delay = int(1000 / 25)
        self.current_output_path = None
        self.frame_count = 0  # 用于越线计数的帧序号

        # ========== 核心组件 ==========
        self.Model = None
        self.deepsort_tracker = None
        self.Obj_Counter = None
        self.class_names = []
        self.yolo5_config = None
        self.is_initialized = False

        # ========== 加载模型 ==========
        self.load_models()

        # ========== 构建界面 ==========
        self.build_ui()

        # ========== 绑定关闭事件 ==========
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_models(self):
        """加载YOLO和DeepSORT模型"""
        try:
            print("=> 正在加载模型，请稍候...")

            class Args:
                def __init__(self):
                    self.weights = 'weights/yolov5s.pt'
                    self.device = 'cpu'
                    self.img_size = 640
                    self.conf_thres = 0.4
                    self.iou_thres = 0.4
                    self.classes = 0
                    self.output = './output'

            self.yolo5_config = Args()

            weights = self.yolo5_config.weights
            if self.yolo5_config.device != "cpu":
                self.Model = torch.load(weights, map_location=lambda storage, loc: storage.cuda(int(self.yolo5_config.device)))['model'].float().fuse().eval()
            else:
                self.Model = torch.load(weights, map_location=torch.device('cpu'))['model'].float().fuse().eval()

            best_torch_version = "1.10.0"
            current_torch_version = str(torch.__version__).split('+')[0]
            if version.parse(current_torch_version) > version.parse(best_torch_version):
                for m in self.Model.modules():
                    if isinstance(m, nn.Upsample):
                        m.recompute_scale_factor = None

            classnames = self.Model.module.names if hasattr(self.Model, 'module') else self.Model.names
            # 默认检测行人
            self.class_names = [classnames[0]]

            cfg = get_config()
            cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
            self.deepsort_tracker = DeepSort(
                cfg.DEEPSORT.REID_CKPT,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE,
                n_init=cfg.DEEPSORT.N_INIT,
                nn_budget=cfg.DEEPSORT.NN_BUDGET,
                use_cuda=False,
                use_appearence=True
            )

            self.Obj_Counter = Object_Counter(self.class_names)
            Global.total_person = set()

            self.is_initialized = True
            print("=> 模型加载完成！")

        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败：{str(e)}")
            self.root.destroy()

    def build_ui(self):
        """构建界面布局"""
        control_frame = tk.Frame(self.root, bg='#f0f0f0', height=110)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        control_frame.pack_propagate(False)

        # 第一行：按钮
        btn_frame1 = tk.Frame(control_frame, bg='#f0f0f0')
        btn_frame1.pack(pady=(5, 2))

        btn_style = {'height': 1, 'width': 12, 'font': ('微软雅黑', 10)}

        self.btn_open = tk.Button(btn_frame1, text="📂 选择视频", command=self.open_video, **btn_style)
        self.btn_open.grid(row=0, column=0, padx=5)

        self.btn_camera = tk.Button(btn_frame1, text="📷 摄像头", command=self.open_camera, **btn_style)
        self.btn_camera.grid(row=0, column=1, padx=5)

        self.btn_start = tk.Button(btn_frame1, text="▶ 开始", command=self.start_detection, **btn_style)
        self.btn_start.grid(row=0, column=2, padx=5)

        self.btn_pause = tk.Button(btn_frame1, text="⏸ 暂停", command=self.pause_detection, state=tk.DISABLED, **btn_style)
        self.btn_pause.grid(row=0, column=3, padx=5)

        self.btn_stop = tk.Button(btn_frame1, text="⏹ 停止", command=self.stop_detection, state=tk.DISABLED, **btn_style)
        self.btn_stop.grid(row=0, column=4, padx=5)

        self.btn_open_folder = tk.Button(btn_frame1, text="📁 打开保存文件夹", command=self.open_output_folder, **btn_style)
        self.btn_open_folder.grid(row=0, column=5, padx=5)

        self.status_label = tk.Label(control_frame, text="状态: 就绪", bg='#f0f0f0', font=('微软雅黑', 10))
        self.status_label.pack(side=tk.LEFT, padx=15)

        # ===== 第二行：滑块 + 复选框 + 下拉菜单 =====
        btn_frame2 = tk.Frame(control_frame, bg='#f0f0f0')
        btn_frame2.pack(pady=(2, 5))

        tk.Label(btn_frame2, text="线位置:", bg='#f0f0f0', font=('微软雅黑', 9)).grid(row=0, column=0, sticky='e', padx=(0, 2))
        self.line_position = tk.IntVar(value=50)
        self.line_scale = tk.Scale(btn_frame2, from_=10, to=90, orient=tk.HORIZONTAL,
                                   variable=self.line_position, length=150,
                                   command=self.update_line_position)
        self.line_scale.grid(row=0, column=1, sticky='w', padx=(0, 10))

        self.line_enabled = tk.BooleanVar(value=True)
        self.line_check = tk.Checkbutton(btn_frame2, text="启用越线计数", variable=self.line_enabled,
                                         bg='#f0f0f0', font=('微软雅黑', 9),
                                         command=self.toggle_line_enabled)
        self.line_check.grid(row=0, column=2, padx=5)

        # ===== 检测类别下拉菜单（含猫） =====
        tk.Label(btn_frame2, text="检测类别:", bg='#f0f0f0', font=('微软雅黑', 9)).grid(row=0, column=3, padx=(15, 2))
        self.detect_class_var = tk.StringVar(value="行人")
        self.detect_class_combo = ttk.Combobox(btn_frame2, textvariable=self.detect_class_var,
                                               values=["行人", "车辆", "猫", "全部"],
                                               state="readonly", width=10)
        self.detect_class_combo.grid(row=0, column=4, padx=5)
        self.detect_class_combo.bind("<<ComboboxSelected>>", self.on_detect_class_changed)

        # 视频显示区域
        video_frame = tk.Frame(self.root, bg='#333333', relief=tk.SUNKEN, bd=2)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.video_label = tk.Label(video_frame, text="请选择视频或打开摄像头\n点击\"开始\"运行检测",
                                    bg='#333333', fg='#aaaaaa',
                                    font=('微软雅黑', 16), justify=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ===== 底部统计信息栏（含方向备注） =====
        info_frame = tk.Frame(self.root, bg='#f0f0f0', height=55)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        info_frame.pack_propagate(False)

        self.info_label = tk.Label(info_frame,
                                   text="当前: 0  累计: 0\n进入(上→下): 0  离开(下→上): 0",
                                   bg='#f0f0f0', font=('微软雅黑', 12, 'bold'),
                                   justify=tk.CENTER)
        self.info_label.pack(pady=8)

    # ===== 检测类别切换相关方法（完整修复版） =====
    def apply_detect_class(self):
        """应用当前选中的检测类别到配置"""
        selected = self.detect_class_var.get()
        class_mapping = {
            "行人": 0,
            "车辆": [2, 3, 5, 7],      # car, motorcycle, bus, truck
            "猫": 15,                   # COCO类别编号15
            "全部": None
        }
        self.yolo5_config.classes = class_mapping[selected]
        
        # ===== 修复：更新类别名称列表 =====
        classnames = self.Model.module.names if hasattr(self.Model, 'module') else self.Model.names
        if selected == "行人":
            self.class_names = [classnames[0]]
        elif selected == "车辆":
            self.class_names = [classnames[2], classnames[3], classnames[5], classnames[7]]
        elif selected == "猫":
            self.class_names = [classnames[15]]
        else:  # 全部
            self.class_names = list(classnames)
        # =====================================
        
        print(f"=> 应用检测类别: {selected} -> classes={self.yolo5_config.classes}")

    def on_detect_class_changed(self, event):
        """检测类别切换时的回调函数"""
        selected = self.detect_class_var.get()
        class_mapping = {
            "行人": 0,
            "车辆": [2, 3, 5, 7],
            "猫": 15,
            "全部": None
        }
        self.yolo5_config.classes = class_mapping[selected]
        
        # ===== 修复：更新类别名称列表 =====
        classnames = self.Model.module.names if hasattr(self.Model, 'module') else self.Model.names
        if selected == "行人":
            self.class_names = [classnames[0]]
        elif selected == "车辆":
            self.class_names = [classnames[2], classnames[3], classnames[5], classnames[7]]
        elif selected == "猫":
            self.class_names = [classnames[15]]
        else:  # 全部
            self.class_names = list(classnames)
        # =====================================
        
        self.status_label.config(text=f"状态: 已切换检测类别 -> {selected}")
        
        if self.is_playing:
            Global.total_person = set()
            if hasattr(self.Obj_Counter, 'line_counter'):
                self.Obj_Counter.line_counter.reset()
            self.update_info(0, 0, 0, 0)
        print(f"=> 检测类别已切换为: {selected} -> class_names={self.class_names[:3]}...")

    # ===== 其他功能方法 =====
    def open_output_folder(self):
        output_dir = os.path.abspath('./output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if os.name == 'nt':
            os.startfile(output_dir)
        else:
            webbrowser.open(output_dir)

    def update_line_position(self, val):
        if hasattr(self.Obj_Counter, 'line_counter'):
            self.Obj_Counter.line_counter.set_line_position(int(val))

    def toggle_line_enabled(self):
        if hasattr(self.Obj_Counter, 'line_counter'):
            self.Obj_Counter.line_counter.enabled = self.line_enabled.get()

    def open_video(self):
        if self.is_playing:
            self.stop_detection()

        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("MP4文件", "*.mp4"), ("AVI文件", "*.avi"), ("所有文件", "*.*")]
        )

        if file_path:
            self.video_path = file_path
            self.is_camera_mode = False
            self.status_label.config(text=f"状态: 已选择视频 - {os.path.basename(file_path)}")
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                self.show_frame_preview(frame)
            cap.release()
            self.btn_start.config(state=tk.NORMAL)
            self.update_info(0, 0, 0, 0)
            Global.total_person = set()
            if hasattr(self.Obj_Counter, 'line_counter'):
                self.Obj_Counter.line_counter.reset()

    def open_camera(self):
        if self.is_playing:
            self.stop_detection()

        self.video_path = 0
        self.is_camera_mode = True
        self.status_label.config(text="状态: 摄像头模式 (按下\"开始\"启动)")
        self.btn_start.config(state=tk.NORMAL)
        self.update_info(0, 0, 0, 0)
        Global.total_person = set()
        if hasattr(self.Obj_Counter, 'line_counter'):
            self.Obj_Counter.line_counter.reset()

        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.show_frame_preview(frame)
            cap.release()

    def show_frame_preview(self, frame):
        try:
            frame = imutils.resize(frame, height=400)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=imgtk, text="")
            self.video_label.image = imgtk
        except Exception as e:
            print(f"预览显示失败: {e}")

    def start_detection(self):
        if not self.is_initialized:
            messagebox.showerror("错误", "模型未加载完成")
            return

        if self.video_path is None:
            messagebox.showwarning("提示", "请先选择视频或打开摄像头")
            return

        if self.is_playing:
            return

        Global.total_person = set()
        if hasattr(self.Obj_Counter, 'line_counter'):
            self.Obj_Counter.line_counter.reset()
            self.Obj_Counter.line_counter.enabled = self.line_enabled.get()
            self.Obj_Counter.line_counter.set_line_position(self.line_position.get())

        self.apply_detect_class()

        self.is_playing = True
        self.is_paused = False
        self.total_frame_count = 0
        self.frame_count = 0
        self.current_output_path = None

        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL, text="⏸ 暂停")
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_open.config(state=tk.DISABLED)
        self.btn_camera.config(state=tk.DISABLED)

        self.status_label.config(text="状态: 检测中...")
        self.update_info(0, 0, 0, 0)

        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()

    def detection_loop(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.root.after(0, self.show_error, "无法打开视频/摄像头")
                return

            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            if self.fps == 0:
                self.fps = 25
            self.delay = int(1000 / self.fps) if self.fps > 0 else 40

            os.makedirs('output', exist_ok=True)
            mkfile_time = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
            output_path = f'./output/gui_result_{mkfile_time}.mp4'
            self.current_output_path = output_path
            self.videowriter = None

            while self.is_playing:
                if self.is_paused:
                    time.sleep(0.1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                self.total_frame_count += 1
                self.frame_count += 1

                try:
                    result_frame, present_num = Counting_Processing(
                        frame,
                        self.yolo5_config,
                        self.Model,
                        self.class_names,
                        self.deepsort_tracker,
                        self.Obj_Counter,
                        isCountPresent=False
                    )

                    if isinstance(result_frame, Exception):
                        raise result_frame

                    current_count = present_num
                    total_count = len(Global.total_person)
                    in_count = 0
                    out_count = 0

                    if hasattr(self.Obj_Counter, 'line_counter'):
                        in_count = self.Obj_Counter.line_counter.count_in
                        out_count = self.Obj_Counter.line_counter.count_out
                        # 尝试从画面中读取当前人数（如果画面有显示的话）
                        # 由于Counting_Processing返回的是处理后的图像，我们可以尝试从Global或outputs中获取
                        # 这里我们使用一个备用方案：从Global中获取

                    self.root.after(0, self.update_info, total_count, current_count, in_count, out_count)
                    self.root.after(0, self.update_video_frame, result_frame)

                    if self.videowriter is None:
                        h, w = result_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                        self.videowriter = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
                    self.videowriter.write(result_frame)

                except Exception as e:
                    print(f"处理帧错误: {e}")
                    continue

                time.sleep(self.delay / 1000.0)

            self.root.after(0, self.on_detection_finish, cap, output_path if self.videowriter else None)

        except Exception as e:
            self.root.after(0, self.show_error, f"检测循环错误: {str(e)}")

    def update_video_frame(self, frame):
        try:
            display_height = 450
            frame_resized = imutils.resize(frame, height=display_height)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=imgtk, text="")
            self.video_label.image = imgtk
        except Exception as e:
            print(f"显示帧失败: {e}")

    def update_info(self, total, current, in_count, out_count):
        """更新底部统计信息（含方向备注）"""
        self.info_label.config(
            text=f"当前: {current}  累计: {total}\n进入(上→下): {in_count}  离开(下→上): {out_count}"
        )

    def show_error(self, msg):
        messagebox.showerror("错误", msg)
        self.stop_detection()

    def on_detection_finish(self, cap, output_path):
        if cap:
            cap.release()
        if self.videowriter:
            self.videowriter.release()
            self.videowriter = None
            self.status_label.config(text=f"状态: 检测完成，视频已保存至 {output_path}")
            self.current_output_path = output_path
        else:
            self.status_label.config(text="状态: 检测结束")

        if self.is_playing:
            self.is_playing = False
            self.btn_start.config(state=tk.NORMAL)
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_open.config(state=tk.NORMAL)
            self.btn_camera.config(state=tk.NORMAL)

    def pause_detection(self):
        if not self.is_playing:
            return

        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="▶ 继续")
            self.status_label.config(text="状态: 已暂停")
        else:
            self.btn_pause.config(text="⏸ 暂停")
            self.status_label.config(text="状态: 检测中...")

    def stop_detection(self):
        self.is_playing = False
        self.is_paused = False

        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="⏸ 暂停")
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.NORMAL)
        self.btn_camera.config(state=tk.NORMAL)

        self.status_label.config(text="状态: 已停止")
        self.video_label.config(image='', text="已停止\n\n请选择视频或打开摄像头")
        self.video_label.image = None

        Global.total_person = set()
        if hasattr(self.Obj_Counter, 'line_counter'):
            self.Obj_Counter.line_counter.reset()
        self.update_info(0, 0, 0, 0)

    def on_closing(self):
        self.is_playing = False
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTrackerApp(root)
    root.mainloop()