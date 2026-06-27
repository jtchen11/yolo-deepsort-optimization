import time
import cv2
import numpy as np  # 新增导入
from self_utils.multi_tasks import Counting_Processing
from self_utils.overall_method import Object_Counter
from deep_sort.configs.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import torch.nn as nn
from packaging import version

# ==================== 方案A ====================
VIDEO_PATH = "car2.mp4"                # 测试视频路径
MODEL_WEIGHT = "weights/yolov5l.pt"    # 方案A：用大模型
USE_APPEARANCE = True          # 方案A：不启用外观特征
DEVICE = "cpu"                         # CPU运行
# =======================================================

print(f"方案A: {MODEL_WEIGHT} + DEEPSORT | 设备: {DEVICE}")

# 1. 加载模型
if DEVICE != "cpu":
    model = torch.load(MODEL_WEIGHT, map_location=lambda storage, loc: storage.cuda(int(DEVICE)))['model'].float().fuse().eval()
else:
    model = torch.load(MODEL_WEIGHT, map_location=torch.device('cpu'))['model'].float().fuse().eval()

if version.parse(torch.__version__.split('+')[0]) > version.parse("1.10.0"):
    for m in model.modules():
        if isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None

classnames = model.module.names if hasattr(model, 'module') else model.names
class_names = list(classnames)

# 2. 加载跟踪器
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
tracker = DeepSort(
    cfg.DEEPSORT.REID_CKPT,
    max_dist=cfg.DEEPSORT.MAX_DIST,
    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg.DEEPSORT.MAX_AGE,
    n_init=cfg.DEEPSORT.N_INIT,
    nn_budget=cfg.DEEPSORT.NN_BUDGET,
    use_cuda=False if DEVICE=="cpu" else True,
    use_appearence=USE_APPEARANCE
)

# 3. 配置参数
class Args:
    pass
yolo5_config = Args()
yolo5_config.device = DEVICE
yolo5_config.img_size = 640
yolo5_config.conf_thres = 0.5
yolo5_config.iou_thres = 0.4
yolo5_config.classes = None
yolo5_config.output = "./output"

obj_counter = Object_Counter(class_names)

# 4. 准备视频写入器（新增！）
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 保存到 output 文件夹，文件名带时间戳防止覆盖
from datetime import datetime
mkfile_time = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
output_video_path = f'./output/benchmark_C_{mkfile_time}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
print(f"结果视频将保存至: {output_video_path}")

# 5. 开始处理（计时只算算法部分，写入不计入FPS，但会同步写入）
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # 核心算法处理
    result_img, _ = Counting_Processing(
        frame, yolo5_config, model, class_names, tracker, obj_counter, isCountPresent=False
    )

    # ===== 修复：检查 result_img 是否为有效的 numpy 数组 =====
    if isinstance(result_img, np.ndarray):
        out_writer.write(result_img)
    else:
        # 如果处理出错（返回了 Exception），则写入原始帧并打印警告
        print(f"警告: 第 {frame_count} 帧处理失败，写入原始帧")
        out_writer.write(frame)

    if frame_count % 30 == 0:
        print(f"  进度: {frame_count}/{total_frames}")

end_time = time.time()
cap.release()
out_writer.release()  # 重要：释放写入器

# 6. 输出结果
elapsed = end_time - start_time
print(f"\n======= 方案A 结果 =======")
print(f"总帧数: {total_frames}")
print(f"总耗时: {elapsed:.2f} 秒")
print(f"平均帧率(FPS): {total_frames / elapsed:.2f}")
print(f"\n✅ 结果视频已保存至: {output_video_path}")
print("请打开这个视频，肉眼观察并统计ID切换次数（目标ID变化的次数）")