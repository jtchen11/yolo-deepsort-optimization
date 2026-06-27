# 基于YOLOv5s+DeepSORT的CPU多目标跟踪系统（含对比实验与GUI）

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.4-green.svg)](https://opencv.org/)

> 在纯CPU（Intel i7，无独显）环境下，通过控制变量实验验证YOLOv5s+DeepSORT为最优方案——帧率达**4.25 FPS**，ID切换仅**5次**，相比YOLOv5l+DeepSORT帧率提升**85.6%**。

---

## 🎯 运行效果

*支持本地视频/USB摄像头输入，实时显示目标ID、类别标签及越线计数（In/Out）*

---

## 📊 对比实验数据（CPU环境：Intel Core i7，无独显）

| 方案 | 检测器 | 跟踪器 | FPS | ID切换 | 结论 |
|------|--------|--------|-----|--------|------|
| A | YOLOv5l | DeepSORT | 2.29 | 3次 | 精度高但卡顿 ❌ |
| B | **YOLOv5s** | **DeepSORT** | **4.25** | **5次** | **✅ 本系统采用** |
| C | YOLOv5l | SORT | 3.93 | 9次 | ID频繁跳变 ❌ |

📁 完整实验数据与复现脚本详见 `models_test/` 文件夹

---

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.8.0
- OpenCV 4.5.4

### 安装与运行

```bash
# 1. 克隆仓库
git clone https://github.com/jtchen11/你的仓库名.git
cd 你的仓库名

# 2. 创建虚拟环境
conda create -n yolo_deepsort_env python=3.8
conda activate yolo_deepsort_env

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载预训练权重到 weights/ 目录
# yolov5s.pt: https://github.com/ultralytics/yolov5/releases
# ckpt.t7: 已包含在仓库中

# 5. 启动系统
python gui.py