import cv2
import numpy as np

class LineCrossingCounter:
    def __init__(self, img_height, img_width, line_position=0.5):
        self.img_height = img_height
        self.img_width = img_width
        self.line_y = int(img_height * line_position)
        # ===== 修复：将缓冲区改为 5 像素（更灵敏） =====
        self.buffer_zone = 0
        self.target_sides = {}
        self.last_seen = {}
        self.count_in = 0
        self.count_out = 0
        self.enabled = True

    def update(self, track_id, bbox, frame_id=0):
        if not self.enabled:
            return False, None

        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)

        # ===== 修复：遗忘机制改为基于帧计数（但因为 frame_id 始终为0，暂时忽略） =====
        if track_id in self.last_seen:
            # 如果间隔超过30帧，重置状态（当 frame_id 正确传入时有效）
            if frame_id - self.last_seen[track_id] > 30:
                if track_id in self.target_sides:
                    del self.target_sides[track_id]
                del self.last_seen[track_id]
        else:
            # 首次出现：初始化侧边状态
            self.target_sides[track_id] = 'above' if center_y < self.line_y else 'below'
            self.last_seen[track_id] = frame_id
            return False, None

        self.last_seen[track_id] = frame_id
        current_side = 'above' if center_y < self.line_y else 'below'
        last_side = self.target_sides.get(track_id, current_side)

        # 如果侧边没变，直接返回
        if current_side == last_side:
            return False, None

        # ===== 修复：跨线检测（降低缓冲区要求，更灵敏） =====
        is_crossing = False
        direction = None

        # 从上往下（进入）
        if last_side == 'above' and current_side == 'below':
            # 只要中心点在线下方超过 buffer_zone 像素就算越线
            if center_y - self.line_y > self.buffer_zone:
                self.count_in += 1
                is_crossing = True
                direction = 'in'
            # 更新状态为 below
            self.target_sides[track_id] = 'below'

        # 从下往上（离开）
        elif last_side == 'below' and current_side == 'above':
            if self.line_y - center_y > self.buffer_zone:
                self.count_out += 1
                is_crossing = True
                direction = 'out'
            # 更新状态为 above
            self.target_sides[track_id] = 'above'

        return is_crossing, direction

    def draw_line_and_info(self, img):
        if not self.enabled:
            return img

        img_height, img_width = img.shape[:2]

        # 画一条主警戒线
        cv2.line(img, (0, self.line_y), (img_width, self.line_y), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "LINE", (10, self.line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        info_text = f"In: {self.count_in}  |  Out: {self.count_out}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(info_text, font, font_scale, thickness)
        text_x = img_width - text_width - 10
        text_y = 60
        cv2.putText(img, info_text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)

        return img

    def set_line_position(self, position_percent):
        self.line_y = int(self.img_height * position_percent / 100)

    def reset(self):
        self.count_in = 0
        self.count_out = 0
        self.target_sides.clear()
        self.last_seen.clear()