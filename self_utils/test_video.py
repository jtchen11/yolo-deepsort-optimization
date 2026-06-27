import cv2

cap = cv2.VideoCapture('my_video.mp4')
print(f"视频是否成功打开: {cap.isOpened()}")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print(f"读取到第 {frame_count} 帧时返回 False")
        break
    if frame is None:
        print(f"第 {frame_count} 帧是 None")
        break
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        print(f"第 {frame_count} 帧尺寸异常: {frame.shape}")
        break
    frame_count += 1
    if frame_count >= 10:  # 只读前10帧
        break

print(f"成功读取 {frame_count} 帧")
cap.release()