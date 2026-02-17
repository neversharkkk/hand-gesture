import cv2
import numpy as np
import time
import pyautogui
from collections import deque
import threading
import math

pyautogui.FAILSAFE = False

MODE_GESTURE = 0
MODE_MOUSE = 1
MODE_ASCII = 2
MODE_COLOR = 3
MODE_SYNTH = 4
current_mode = MODE_GESTURE

gesture_history = deque(maxlen=5)

# 全局缩放因子（用于适配不同分辨率）
scale_factor = 1.0

# 背景减除器
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60, detectShadows=False)

# 手部位置历史（用于平滑）- 减少历史长度
hand_position_history = deque(maxlen=3)
hand_size_history = deque(maxlen=3)

# 上一次有效的轮廓
last_valid_contour = None
last_valid_time = 0

# 音乐合成器类
class AudioSynthesizer:
    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.running = False
        self.thread = None
        
        # 主振荡器参数
        self.frequency = 440.0
        self.frequency2 = 550.0  # 第二振荡器
        self.detune = 0.0  # 失谐
        
        # LFO参数
        self.lfo_rate = 1.0
        self.lfo_depth = 0.3
        self.lfo_type = 0  # 0=sin, 1=square, 2=triangle
        
        # 包络参数
        self.attack = 0.05
        self.decay = 0.1
        self.sustain = 0.7
        self.release = 0.2
        self.envelope = 0.0
        self.envelope_target = 0.0
        
        # 滤波器参数
        self.filter_cutoff = 1000.0
        self.filter_resonance = 0.5
        
        # 效果参数
        self.reverb_mix = 0.3
        self.delay_time = 0.3
        
        # 相位
        self.phase = 0.0
        self.phase2 = 0.0
        self.lfo_phase = 0.0
        
        # 颜色影响
        self.color_hue = 0.0
        self.color_sat = 0.0
        
        # 音量
        self.volume = 0.4
        
        # 历史缓冲（用于延迟效果）
        self.delay_buffer = np.zeros(int(sample_rate * 0.5))
        self.delay_index = 0
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.3)
    
    def _audio_loop(self):
        try:
            import sounddevice as sd
            with sd.OutputStream(samplerate=self.sample_rate, 
                                channels=1, 
                                callback=self._audio_callback,
                                blocksize=self.buffer_size):
                while self.running:
                    time.sleep(0.005)
        except ImportError:
            print("sounddevice not installed, audio disabled")
            self.running = False
    
    def _audio_callback(self, outdata, frames, time_info, status):
        t = np.arange(frames) / self.sample_rate
        
        # LFO生成
        if self.lfo_type == 0:
            lfo = np.sin(2 * np.pi * self.lfo_rate * (self.lfo_phase + t))
        elif self.lfo_type == 1:
            lfo = np.sign(np.sin(2 * np.pi * self.lfo_rate * (self.lfo_phase + t)))
        else:
            lfo = 2 * np.abs(2 * (self.lfo_phase + t) * self.lfo_rate % 1 - 0.5) - 1
        
        lfo_mod = 1.0 + lfo * self.lfo_depth
        
        # 主振荡器
        freq1 = self.frequency * lfo_mod * (1.0 + self.color_hue * 0.1)
        phase_inc1 = 2 * np.pi * freq1 / self.sample_rate
        self.phase = self.phase + np.cumsum(phase_inc1)
        osc1 = np.sin(self.phase)
        
        # 第二振荡器（失谐）
        freq2 = self.frequency2 * (1.0 + self.detune * 0.1) * lfo_mod
        phase_inc2 = 2 * np.pi * freq2 / self.sample_rate
        self.phase2 = self.phase2 + np.cumsum(phase_inc2)
        osc2 = np.sin(self.phase2)
        
        # 混合振荡器
        wave = osc1 * 0.6 + osc2 * 0.4
        
        # 添加泛音
        wave += np.sin(self.phase * 2) * 0.2 * self.color_sat
        wave += np.sin(self.phase * 3) * 0.1 * self.color_sat
        
        # 简单低通滤波
        cutoff_mod = self.filter_cutoff * (1.0 + lfo * 0.3)
        filter_coef = np.exp(-2 * np.pi * cutoff_mod / self.sample_rate)
        
        # 包络处理
        envelope_diff = self.envelope_target - self.envelope
        if abs(envelope_diff) > 0.001:
            rate = 1.0 / self.attack if envelope_diff > 0 else 1.0 / self.release
            self.envelope += np.sign(envelope_diff) * min(abs(envelope_diff), rate / self.sample_rate * frames)
        
        wave = wave * self.envelope * self.volume
        
        # 延迟效果
        delay_samples = int(self.delay_time * self.sample_rate)
        for i in range(frames):
            delayed = self.delay_buffer[self.delay_index] * 0.3
            wave[i] = wave[i] + delayed
            self.delay_buffer[self.delay_index] = wave[i]
            self.delay_index = (self.delay_index + 1) % len(self.delay_buffer)
        
        # 更新相位
        self.phase = self.phase[-1] % (2 * np.pi * 1000)
        self.phase2 = self.phase2[-1] % (2 * np.pi * 1000)
        self.lfo_phase = (self.lfo_phase + self.lfo_rate * frames / self.sample_rate) % 1.0
        
        outdata[:, 0] = np.clip(wave, -1, 1).astype(np.float32)
    
    def update_from_gesture(self, hand_y, hand_area, hand_x, colors, finger_count=0):
        """从手势和颜色更新合成器参数，让声音符合画面"""
        # 手的Y位置控制频率 (高=高频，低=低频)
        self.frequency = 80 + (1.0 - hand_y) * 700  # 80-780 Hz
        self.frequency2 = self.frequency * 1.5  # 五度
        
        # 手的面积控制LFO深度和包络
        self.lfo_depth = min(0.6, hand_area * 0.0008)
        self.envelope_target = min(1.0, hand_area * 0.0015)
        
        # 手的X位置控制LFO速率和滤波器
        self.lfo_rate = 0.3 + hand_x * 4.0  # 0.3-4.3 Hz
        self.filter_cutoff = 200 + hand_x * 2000  # 200-2200 Hz
        
        # 手指数量控制失谐和LFO类型
        self.detune = finger_count * 2.0
        self.lfo_type = finger_count % 3
        
        # 颜色影响 - 让声音符合画面
        if colors is not None and len(colors) > 0:
            # 计算主色调的HSV
            main_color = colors[0]
            r, g, b = main_color[0] / 255.0, main_color[1] / 255.0, main_color[2] / 255.0
            
            # 计算色相 (Hue) 0-360度
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            delta = max_c - min_c
            
            if delta == 0:
                hue = 0
            elif max_c == r:
                hue = 60 * (((g - b) / delta) % 6)
            elif max_c == g:
                hue = 60 * (((b - r) / delta) + 2)
            else:
                hue = 60 * (((r - g) / delta) + 4)
            
            # 计算饱和度 (Saturation)
            saturation = delta / max_c if max_c > 0 else 0
            
            # 计算明度 (Value/Brightness)
            value = max_c
            
            # 色相影响：频率微调（不同色相产生不同音色感觉）
            # 红(0°) -> 低频偏移，蓝(240°) -> 高频偏移
            hue_factor = (hue / 360.0) * 0.25  # ±25%频率偏移（增加影响）
            self.frequency = self.frequency * (1.0 + hue_factor - 0.125)
            
            # 饱和度影响：泛音丰富度和LFO深度
            # 高饱和度 -> 更丰富的泛音，更深的LFO
            self.color_sat = saturation
            self.lfo_depth = min(0.7, self.lfo_depth + saturation * 0.2)  # 增加影响
            
            # 明度影响：音量和滤波器
            # 高明度 -> 更亮的声音（更高的滤波器截止）
            self.filter_cutoff = self.filter_cutoff * (0.6 + value * 0.8)  # 增加影响
            self.volume = 0.3 + value * 0.3  # 明度影响音量
            
            # 存储用于显示
            self.color_hue = hue / 360.0
            
            # 颜色多样性：如果有多种颜色，增加声音复杂度
            if len(colors) >= 3:
                # 计算颜色多样性
                color_variance = np.var(colors[:4], axis=0).mean() / 255.0
                self.detune += color_variance * 5  # 增加失谐
                self.reverb_mix = 0.2 + color_variance * 0.4  # 增加混响
        
        # 延迟时间
        self.delay_time = 0.1 + hand_x * 0.3

# 全局合成器实例
synth = None

def smooth_gesture(gesture_name):
    if gesture_name:
        gesture_history.append(gesture_name)
    
    if len(gesture_history) >= 3:
        from collections import Counter
        counts = Counter(gesture_history)
        most_common = counts.most_common(1)[0]
        # 只需要超过一半就返回
        if most_common[1] >= 2:
            return most_common[0]
    
    return gesture_name if gesture_name else ""

def smooth_position(x, y, w, h):
    """平滑手部位置 - 更快响应"""
    global hand_position_history, hand_size_history
    
    hand_position_history.append((x, y))
    hand_size_history.append((w, h))
    
    # 简单平均，更快响应
    if len(hand_position_history) == 0:
        return x, y, w, h
    
    sum_x = sum(p[0] for p in hand_position_history)
    sum_y = sum(p[1] for p in hand_position_history)
    sum_w = sum(s[0] for s in hand_size_history)
    sum_h = sum(s[1] for s in hand_size_history)
    n = len(hand_position_history)
    
    return sum_x // n, sum_y // n, sum_w // n, sum_h // n

def detect_skin(image):
    """多颜色空间融合的皮肤检测"""
    # HSV空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 15, 40], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # YCrCb空间
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # LAB空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lower_lab = np.array([0, 130, 130], dtype=np.uint8)
    upper_lab = np.array([255, 160, 160], dtype=np.uint8)
    mask_lab = cv2.inRange(lab, lower_lab, upper_lab)
    
    # 融合三个颜色空间的结果
    mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    mask = cv2.bitwise_or(mask, mask_lab)
    
    return mask

def process_hand(image):
    """改进的手部检测算法"""
    global last_valid_contour, last_valid_time
    
    h, w = image.shape[:2]
    
    # 1. 皮肤颜色检测
    skin_mask = detect_skin(image)
    
    # 2. 背景减除
    fg_mask = bg_subtractor.apply(image, learningRate=0.001)
    fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
    
    # 3. 融合检测结果
    combined = cv2.bitwise_and(skin_mask, fg_mask)
    
    # 如果融合结果太少，使用纯皮肤检测
    min_combined_pixels = int(1000 * (scale_factor ** 2))
    if cv2.countNonZero(combined) < min_combined_pixels:
        combined = skin_mask
    
    # 4. 形态学处理（根据分辨率调整内核大小）
    kernel_size = max(5, int(7 * scale_factor))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 5. 查找轮廓
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # 如果没有检测到轮廓，检查是否可以使用上一次的有效轮廓
        current_time = time.time()
        if last_valid_contour is not None and current_time - last_valid_time < 0.5:
            return last_valid_contour, None, combined
        return None, None, combined
    
    # 6. 筛选手部轮廓
    best_contour = None
    best_score = 0
    
    # 根据分辨率缩放阈值
    min_area = int(1500 * (scale_factor ** 2))
    area_low = int(2000 * (scale_factor ** 2))
    area_high = int(80000 * (scale_factor ** 2))
    area_ext_low = int(1500 * (scale_factor ** 2))
    area_ext_high = int(100000 * (scale_factor ** 2))
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = cw / ch if ch > 0 else 0
        
        # 计算得分
        score = 0
        
        # 面积得分 - 放宽范围（使用缩放后的阈值）
        if area_low < area < area_high:
            score += 30
        elif area_ext_low < area < area_ext_high:
            score += 20
        
        # 圆形度得分 - 放宽范围
        if 0.2 < circularity < 0.7:
            score += 25
        elif 0.15 < circularity < 0.8:
            score += 15
        
        # 实度得分
        if 0.4 < solidity < 0.95:
            score += 25
        elif 0.3 < solidity < 0.98:
            score += 15
        
        # 长宽比得分
        if 0.5 < aspect_ratio < 2.0:
            score += 20
        elif 0.3 < aspect_ratio < 2.5:
            score += 10
        
        # 如果有历史位置，给予位置接近的轮廓额外分数
        if len(hand_position_history) > 0:
            last_x, last_y = hand_position_history[-1]
            last_w, last_h = hand_size_history[-1]
            center_x = x + cw // 2
            center_y = y + ch // 2
            last_center_x = last_x + last_w // 2
            last_center_y = last_y + last_h // 2
            
            distance = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
            max_distance = np.sqrt(w**2 + h**2) * 0.4  # 放宽最大允许移动距离
            
            if distance < max_distance:
                score += int(30 * (1 - distance / max_distance))
        
        if score > best_score:
            best_score = score
            best_contour = cnt
    
    if best_contour is not None:
        last_valid_contour = best_contour
        last_valid_time = time.time()
    
    defects = None
    if best_contour is not None:
        hull = cv2.convexHull(best_contour, returnPoints=False)
        defects = cv2.convexityDefects(best_contour, hull)
    
    return best_contour, defects, combined

def recognize_gesture(contour, defects):
    if contour is None:
        return "", 0
    
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    solidity = area / hull_area if hull_area > 0 else 0
    
    x, y, cw, ch = cv2.boundingRect(contour)
    
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # 计算手指数量 - 使用凸包顶点检测
    finger_count = 0
    
    if defects is not None and defects.shape[0] > 0:
        # 获取凸包点
        hull_points = cv2.convexHull(contour, returnPoints=True)
        
        # 计算手的中心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + cw // 2, y + ch // 2
        
        # 找出凸包的顶点（手指尖）
        finger_tips = []
        
        for i, pt in enumerate(hull_points):
            pt = pt[0]
            
            # 只考虑手部上半部分的点
            if pt[1] < cy + ch * 0.1:
                # 计算该点与中心的距离
                dist = np.sqrt((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2)
                
                # 如果距离足够远，可能是手指尖
                if dist > ch * 0.25:
                    finger_tips.append(pt)
        
        # 过滤相邻的重复点
        if len(finger_tips) > 0:
            filtered_tips = [finger_tips[0]]
            for tip in finger_tips[1:]:
                # 检查是否与已有点太近
                is_duplicate = False
                for ft in filtered_tips:
                    if np.sqrt((tip[0] - ft[0]) ** 2 + (tip[1] - ft[1]) ** 2) < cw * 0.2:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    filtered_tips.append(tip)
            
            finger_count = len(filtered_tips)
        
        # 使用缺陷点验证手指数量
        valid_defects = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > 400:  # 深度阈值
                cnt_f = tuple(contour[f][0])
                # 缺陷点应该在手的下半部分
                if cnt_f[1] > cy - ch * 0.1:
                    valid_defects += 1
        
        # 手指数量 = 缺陷数量 + 1（如果缺陷检测更可靠）
        if valid_defects > 0 and valid_defects < 5:
            finger_count = valid_defects + 1
        
        finger_count = min(max(finger_count, 0), 5)
    
    # 手势判断
    gesture_name = ""
    
    # 握拳：实度低
    if solidity < 0.35:
        gesture_name = "Fist"
    # 手掌：实度高或手指多
    elif solidity > 0.75 or finger_count >= 4:
        gesture_name = "Paper"
    # 剪刀：2个手指
    elif finger_count == 2:
        gesture_name = "Victory"
    # 一：1个手指
    elif finger_count == 1:
        gesture_name = "One"
    # 三：3个手指
    elif finger_count == 3:
        gesture_name = "Three"
    # 默认
    else:
        gesture_name = "Point"
    
    gesture_name = smooth_gesture(gesture_name)
    
    return gesture_name, finger_count

def draw_3d_cube(image, center_x, center_y, scale, rot_x, rot_y):
    h, w = image.shape[:2]
    size = int(60 * scale)
    
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ]) * size
    
    angle_x = np.radians(rot_x)
    angle_y = np.radians(rot_y)
    
    Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    vertices = vertices @ Ry.T @ Rx.T
    
    vertices[:, 0] += center_x
    vertices[:, 1] += center_y
    
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    
    faces = [
        ([0,1,2,3], (200, 100, 100)),
        ([4,5,6,7], (100, 200, 100)),
        ([0,1,5,4], (100, 100, 200)),
        ([2,3,7,6], (200, 200, 100)),
        ([1,2,6,5], (100, 200, 200)),
        ([0,3,7,4], (200, 100, 200))
    ]
    
    camera_pos = np.array([0, 0, 500])
    
    for face_indices, color in faces:
        face_center = np.mean(vertices[face_indices], axis=0)
        v1 = vertices[face_indices[1]] - vertices[face_indices[0]]
        v2 = vertices[face_indices[2]] - vertices[face_indices[0]]
        normal = np.cross(v1, v2)
        view_vec = camera_pos - face_center
        if np.dot(normal, view_vec) > 0:
            pts = vertices[face_indices].reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(image, [pts], color)
    
    for edge in edges:
        pt1 = tuple(vertices[edge[0]][:2].astype(int))
        pt2 = tuple(vertices[edge[1]][:2].astype(int))
        cv2.line(image, pt1, pt2, (50, 50, 50), 2)
    
    return image

def control_cube(gesture_name, finger_count, hand_center, w, h):
    global cube_rotation, cube_scale, cube_auto_rotate
    
    if hand_center is None:
        return
    
    cx, cy = hand_center
    
    if gesture_name == "One":
        cube_auto_rotate = False
        dx = (cx - w // 2) / w
        dy = (cy - h // 2) / h
        cube_rotation[0] += dy * 5
        cube_rotation[1] += dx * 5
    
    elif gesture_name == "Victory":
        cube_auto_rotate = False
        dy = (cy - h // 2) / h
        cube_scale = max(0.3, min(2.5, cube_scale - dy * 0.05))
    
    elif gesture_name == "Paper":
        cube_auto_rotate = True

# ASCII艺术字符集（按亮度排序，增加多样性）
ASCII_CHARS = " .',:;ilI1|\\/()[]{}?_-+~=<>!@#$%&*#"
ASCII_CHARS_EXTENDED = " .',:;ilI1|\\/()[]{}?_-+~=<>!@#$%&*#O0O8&@#*+"  # 扩展字符集（更多基础字符）
ASCII_COLORS = [
    (50, 50, 50),
    (70, 70, 70),
    (90, 90, 90),
    (110, 110, 110),
    (130, 130, 130),
    (150, 150, 150),
    (170, 170, 170),
    (190, 190, 190),
    (210, 210, 210),
    (230, 230, 230),
    (255, 255, 255),
    (255, 255, 150),
    (255, 230, 100),
    (255, 200, 50),
]

def create_ascii_art(image, contour, mask, synth=None):
    """在检测到的手部区域创建ASCII艺术效果，符号颜色根据明暗渐变，合成器影响字符"""
    if contour is None:
        return image
    
    h, w = image.shape[:2]
    
    # 获取手部边界框
    x, y, cw, ch = cv2.boundingRect(contour)
    
    # 扩展边界框
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    cw = min(w - x, cw + 2 * padding)
    ch = min(h - y, ch + 2 * padding)
    
    # 裁剪手部区域
    hand_region = image[y:y+ch, x:x+cw].copy()
    mask_region = mask[y:y+ch, x:x+cw]
    
    # 转换为灰度图
    gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
    
    # 根据模式设置不同参数
    if synth is not None:
        # 模式5：合成器模式，更大更稀疏，扩展字符集，无震荡
        base_char_size = max(14, min(20, cw // 12))
        # 频率影响字符大小：更灵敏（高频=小字符，低频=大字符）
        freq_factor = 1.0 - (synth.frequency - 80) / 700 * 0.5  # 更灵敏
        char_size = int(base_char_size * freq_factor)
        char_size = max(8, min(28, char_size))  # 扩大范围
        jitter_freq = 0  # 无震荡
        char_offset = int(synth.envelope_target * 30)  # 更多样的字符偏移
        chars_to_use = ASCII_CHARS_EXTENDED  # 扩展字符集
        skip_rate = 0.25  # 更稀疏
    else:
        # 模式3：普通ASCII模式，回到之前的大小
        char_size = max(4, min(8, cw // 30))
        jitter_freq = 0
        char_offset = 0
        chars_to_use = ASCII_CHARS
        skip_rate = 0
    
    # 创建ASCII艺术图像
    ascii_h = ch // char_size
    ascii_w = cw // char_size
    
    if ascii_h < 3 or ascii_w < 3:
        return image
    
    # 创建输出图像
    result = image.copy()
    
    # 当前时间用于抖动
    current_time = time.time()
    
    # 为每个字符生成随机相位偏移（基于位置，保持一致性）
    np.random.seed(42)
    phase_offsets = np.random.random((ascii_h, ascii_w)) * 2 * np.pi
    
    # 在手部区域绘制ASCII字符（稀疏采样）
    for row in range(ascii_h):
        for col in range(ascii_w):
            # 随机跳过一些字符，让显示更稀疏多样
            if skip_rate > 0 and np.random.random() < skip_rate:
                continue
            
            # 计算当前块的平均亮度
            y_start = row * char_size
            y_end = min((row + 1) * char_size, ch)
            x_start = col * char_size
            x_end = min((col + 1) * char_size, cw)
            
            block = gray[y_start:y_end, x_start:x_end]
            block_mask = mask_region[y_start:y_end, x_start:x_end]
            
            # 只处理在手部区域内的块
            if cv2.countNonZero(block_mask) < (block_mask.size * 0.3):
                continue
            
            if block.size == 0:
                continue
            
            # 计算平均亮度
            avg_brightness = np.mean(block)
            
            # 合成器抖动影响（模式5无震荡）
            brightness_jitter = avg_brightness
            
            # 映射到ASCII字符
            base_char_idx = int(brightness_jitter / 255 * (len(chars_to_use) - 1))
            
            # 模式5：多种字符显示，种类随合成器参数变化
            if synth is not None:
                # 基于位置的多样性（使用模运算确保在有效范围内）
                pos_variety = (row * 7 + col * 11 + int(phase_offsets[row, col] * 100)) % 8
                
                # 合成器参数影响字符种类（更灵敏）
                freq_variety = int((synth.frequency - 80) / 50) % 6  # 频率影响
                lfo_variety = int(synth.lfo_rate * 2) % 5  # LFO速率影响
                envelope_variety = int(synth.envelope_target * 10) % 4  # 包络影响
                
                # 组合偏移（确保多样性且不会超出范围太多）
                total_variety = pos_variety + freq_variety + lfo_variety + envelope_variety
                
                # 使用模运算确保字符种类多样性
                char_idx = (base_char_idx + total_variety) % len(chars_to_use)
            else:
                char_idx = base_char_idx
            
            char_idx = max(0, min(len(chars_to_use) - 1, char_idx))
            
            # 蓝色渐变：根据亮度从深蓝到浅蓝
            blue_val = int(80 + brightness_jitter * 0.68)
            green_val = int(40 + brightness_jitter * 0.55)
            red_val = int(10 + brightness_jitter * 0.35)
            gradient_color = (blue_val, green_val, red_val)
            
            # 获取字符
            char = chars_to_use[char_idx]
            
            # 计算绘制位置
            draw_x = x + x_start + char_size // 2
            draw_y = y + y_start + char_size
            
            # 绘制ASCII字符
            font_scale = char_size / 18 if synth is not None else char_size / 20
            cv2.putText(result, char, (draw_x, draw_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       gradient_color, 1, cv2.LINE_AA)
    
    return result

def extract_8_dominant_colors(image):
    """提取图像中像素数量最多的8种主色调（极速版）"""
    h, w = image.shape[:2]
    
    # 缩小图像以加快处理速度
    small = cv2.resize(image, (w // 12, h // 12))
    
    # 转换为RGB
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    
    # 重塑为像素列表
    pixels = rgb.reshape(-1, 3)
    
    # 随机采样以加快速度
    if len(pixels) > 500:
        indices = np.random.choice(len(pixels), 500, replace=False)
        pixels = pixels[indices]
    
    # 转换为float32
    pixels = np.float32(pixels)
    
    # K-means聚类 - 极速参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 3.0)
    _, labels, centers = cv2.kmeans(pixels, 8, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    
    # 统计每个聚类的像素数量
    unique, counts = np.unique(labels, return_counts=True)
    
    # 按像素数量排序（从多到少）
    sorted_indices = np.argsort(-counts)
    
    # 转换为整数并按像素数量排序
    centers = np.uint8(centers)
    sorted_centers = centers[sorted_indices]
    
    return sorted_centers

def smooth_color_centers(new_centers, history):
    """平滑颜色采样，减缓颜色过渡速度"""
    if not history:
        return new_centers
    
    # 将历史颜色转换为数组
    history_array = np.array(list(history), dtype=np.float32)  # (n, 8, 3)
    new_array = np.array(new_centers, dtype=np.float32)  # (8, 3)
    
    # 计算平滑后的颜色
    smoothed = np.zeros_like(new_array)
    
    for i in range(8):
        current = new_array[i]
        
        # 在历史中找到最相近的颜色
        min_dist = float('inf')
        best_match = current.copy()
        
        for hist in history_array:
            for j in range(8):
                dist = np.sum((hist[j] - current) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = hist[j]
        
        # 平滑混合：50%新颜色 + 50%历史颜色，减缓过渡
        if min_dist < 8000:  # 放宽匹配阈值
            smoothed[i] = current * 0.5 + best_match * 0.5
        else:
            smoothed[i] = current
    
    return np.uint8(smoothed)

def create_soft_color_palette(centers):
    """根据主色调创建生动鲜艳的调色板"""
    palette = []
    
    # 明暗梯度 - 保持层次感
    factors = [0.30, 0.42, 0.54, 0.66, 0.78, 0.90, 1.02, 1.15]
    
    for color in centers:
        r, g, b = color
        
        # 计算原始颜色的亮度和饱和度
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        saturation = (max_val - min_val) / max_val if max_val > 0 else 0
        
        for factor in factors:
            # 根据亮度调整
            if luminance < 80:
                # 暗色系：提升亮度
                adjusted_factor = factor * 1.15 + 0.1
            elif luminance < 150:
                # 中间调：保持层次
                adjusted_factor = factor * 1.05 + 0.05
            else:
                # 亮色系：保持高光
                adjusted_factor = factor
            
            # 大幅增强饱和度，保持画面生动
            if saturation < 0.5:
                # 低饱和度：大幅提升
                sat_boost = 1.3
            elif saturation < 0.7:
                # 中等饱和度：适度提升
                sat_boost = 1.15
            else:
                # 高饱和度：保持
                sat_boost = 1.0
            
            # 应用调整
            new_r = int(min(255, max(0, r * adjusted_factor)))
            new_g = int(min(255, max(0, g * adjusted_factor)))
            new_b = int(min(255, max(0, b * adjusted_factor)))
            
            # 增强饱和度
            gray_val = 0.299 * new_r + 0.587 * new_g + 0.114 * new_b
            new_r = int(gray_val + (new_r - gray_val) * sat_boost)
            new_g = int(gray_val + (new_g - gray_val) * sat_boost)
            new_b = int(gray_val + (new_b - gray_val) * sat_boost)
            
            # 确保在有效范围内
            new_r = min(255, max(0, new_r))
            new_g = min(255, max(0, new_g))
            new_b = min(255, max(0, new_b))
            
            palette.append((new_b, new_g, new_r))  # BGR格式
    
    return palette

def apply_soft_color_mapping_fast(image, centers):
    """应用生动鲜艳的色彩映射到图像（含输出滤波和色彩增强）"""
    h, w = image.shape[:2]
    
    # 创建调色板
    palette = create_soft_color_palette(centers)
    palette = np.array(palette, dtype=np.uint8)
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 转换为RGB用于颜色匹配
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    # 预计算距离矩阵 - 使用广播优化
    centers_array = np.array(centers, dtype=np.float32)  # (8, 3)
    
    # 重塑为二维数组
    pixels = rgb.reshape(-1, 3)  # (N, 3)
    
    # 使用广播计算距离 - 更高效
    diff = pixels[:, np.newaxis, :] - centers_array[np.newaxis, :, :]  # (N, 8, 3)
    distances = np.sum(diff ** 2, axis=2)  # (N, 8)
    
    # 找到最近的主色调索引
    color_indices = np.argmin(distances, axis=1)
    
    # 计算明度级别
    gray_flat = gray.reshape(-1).astype(np.float32)
    levels = (gray_flat / 255 * 7).astype(np.int32)
    levels = np.clip(levels, 0, 7)
    
    # 计算调色板索引
    palette_indices = color_indices * 8 + levels
    
    # 应用调色板
    result_flat = palette[palette_indices]
    result = result_flat.reshape(h, w, 3)
    
    # 输出滤波：轻微高斯模糊减少噪点
    result = cv2.GaussianBlur(result, (3, 3), 0.5)
    
    # 色彩增强：提升对比度和饱和度
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)  # 饱和度+15%
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)  # 亮度+5%
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return result

def draw_color_palette_ui(image, centers):
    """绘制颜色调色板UI"""
    h, w = image.shape[:2]
    
    # 在右上角绘制调色板
    num_colors = len(centers)
    palette_h = 15
    palette_w = 20
    start_x = w - (num_colors * palette_w + 10)
    start_y = 70
    
    # 绘制背景
    cv2.rectangle(image, (start_x - 5, start_y - 5), 
                 (w - 5, start_y + palette_h + 5), (0, 0, 0), -1)
    
    # 绘制每种颜色
    for i, color in enumerate(centers):
        px = start_x + i * palette_w
        # color是RGB格式，转换为BGR
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.rectangle(image, (px, start_y), (px + palette_w - 1, start_y + palette_h), bgr, -1)
    
    return image

def draw_ui(image, fps, current_mode, color_centers=None, synth_params=None):
    h, w = image.shape[:2]
    
    font_scale = 0.5 if h < 600 else 0.85
    font_scale_small = 0.4 if h < 600 else 0.6
    
    # 顶部UI
    cv2.rectangle(image, (0, 0), (w, int(h * 0.08)), (0, 0, 0), -1)
    cv2.putText(image, 'Hand Gesture', (10, int(h * 0.055)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    cv2.putText(image, f'FPS:{int(fps)}', (w - 80, int(h * 0.055)), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
    
    # 合成器模式特殊UI - 信息显示在顶部
    if current_mode == MODE_SYNTH and synth_params is not None:
        freq, lfo_rate, lfo_depth, detune, lfo_type, hue, sat, val = synth_params
        lfo_names = ['Sin', 'Sqr', 'Tri']
        # 在顶部右侧显示合成器参数
        info_x = w - 300
        cv2.putText(image, f'Freq:{int(freq)}Hz Detune:{detune:.1f}', (info_x, int(h * 0.035)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 200, 100), 1)
        cv2.putText(image, f'LFO:{lfo_rate:.1f}Hz {lfo_names[lfo_type]} D:{lfo_depth:.2f}', (info_x, int(h * 0.065)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (100, 255, 200), 1)
    
    # 底部UI
    cv2.rectangle(image, (0, int(h * 0.92)), (w, h), (0, 0, 0), -1)
    cv2.putText(image, 'q:Quit 1:Mouse 2:Gesture 3:ASCII 4:Color 5:Synth r:Reset', (10, int(h * 0.96)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (180, 180, 180), 1)
    
    if current_mode == MODE_ASCII:
        cv2.putText(image, 'ASCII Art Mode', (10, int(h * 0.90)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 200, 100), 1)
    elif current_mode == MODE_COLOR:
        cv2.putText(image, '8-Color Mode', (10, int(h * 0.90)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (200, 100, 255), 1)
        if color_centers is not None:
            image = draw_color_palette_ui(image, color_centers)
    elif current_mode == MODE_SYNTH:
        cv2.putText(image, 'Synthesizer Mode', (10, int(h * 0.90)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 100, 100), 1)
    else:
        mode_color = (0, 255, 0) if current_mode == MODE_GESTURE else (0, 150, 255)
        mode_text = 'Gesture' if current_mode == MODE_GESTURE else 'Mouse'
        cv2.putText(image, f'Mode: {mode_text}', (10, int(h * 0.90)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, mode_color, 1)
    
    return image

def main():
    global current_mode
    
    print("Initializing...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # 尝试设置高分辨率，然后获取摄像头实际支持的规格
    # 先尝试常见的高分辨率
    resolutions_to_try = [
        (1920, 1080),
        (1280, 720),
        (640, 480),
    ]
    
    for width, height in resolutions_to_try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w >= width * 0.9:  # 如果实际分辨率接近目标
            break
    
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 获取摄像头实际规格
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Camera specs: {actual_w}x{actual_h} @ {actual_fps}fps")
    
    # 根据分辨率自动适配参数
    global scale_factor
    if actual_w >= 1280:
        # 高分辨率：使用更大的处理区域
        scale_factor = actual_w / 640
        print(f"High resolution detected, scale factor: {scale_factor:.2f}")
    else:
        scale_factor = 1.0
    
    print("Starting... Press 'q' to quit or click X to close")
    
    prev_time = 0
    fps_smooth = 30
    
    # 颜色模式变量
    color_centers = None
    color_history = deque(maxlen=10)
    
    # 合成器模式变量
    global synth
    synth_params = None
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        image = cv2.flip(image, 1)
        
        h, w = image.shape[:2]
        
        contour, defects, mask = process_hand(image)
        
        display_image = image.copy()
        
        gesture_name, finger_count = recognize_gesture(contour, defects)
        
        hand_center = None
        
        if contour is not None:
            # 手势模式：只勾勒手部线条
            if current_mode == MODE_GESTURE:
                cv2.drawContours(display_image, [contour], 0, (0, 255, 0), 2)
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # 使用平滑位置
            smooth_x, smooth_y, smooth_cw, smooth_ch = smooth_position(x, y, cw, ch)
            
            hand_center = (smooth_x + smooth_cw // 2, smooth_y + smooth_ch // 2)
            
            if current_mode == MODE_GESTURE:
                if gesture_name:
                    cv2.putText(display_image, gesture_name, (smooth_x, smooth_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            elif current_mode == MODE_MOUSE:
                if hand_center:
                    screen_width, screen_height = pyautogui.size()
                    screen_x = int((hand_center[0] / w) * screen_width)
                    screen_y = int((hand_center[1] / h) * screen_height)
                    pyautogui.moveTo(screen_x, screen_y, duration=0.05)
                    cv2.circle(display_image, hand_center, 8, (0, 200, 255), -1)
        
        # ASCII艺术模式
        if current_mode == MODE_ASCII and contour is not None:
            display_image = create_ascii_art(display_image, contour, mask)
        
        # 颜色调色板模式（8色）
        if current_mode == MODE_COLOR:
            # 每帧更新颜色（30Hz），实时响应
            new_colors = extract_8_dominant_colors(display_image)
            
            # 平滑颜色采样
            color_centers = smooth_color_centers(new_colors, color_history)
            
            # 更新历史
            color_history.append(color_centers.copy())
            
            # 应用生动鲜艳色彩映射
            display_image = apply_soft_color_mapping_fast(display_image, color_centers)
        
        # 合成器模式
        if current_mode == MODE_SYNTH:
            # 启动合成器
            if synth is None:
                synth = AudioSynthesizer()
                synth.start()
            
            # 使用ASCII艺术作为背景（传入合成器影响字符）
            if contour is not None:
                # 先更新合成器参数
                x, y, cw, ch = cv2.boundingRect(contour)
                hand_y = y / h
                hand_x = x / w
                hand_area = cw * ch
                
                new_colors = extract_8_dominant_colors(display_image)
                synth.update_from_gesture(hand_y, hand_area, hand_x, new_colors, finger_count)
                
                # 然后创建受合成器影响的ASCII艺术
                display_image = create_ascii_art(display_image, contour, mask, synth)
                
                # 显示参数
                synth_params = (synth.frequency, synth.lfo_rate, synth.lfo_depth, synth.detune, synth.lfo_type, synth.color_hue, synth.color_sat, synth.envelope_target)
            else:
                # 没有检测到手时，降低音量
                if synth is not None:
                    synth.envelope_target = 0.0
        
        # 停止合成器（切换模式时）
        if current_mode != MODE_SYNTH and synth is not None:
            synth.stop()
            synth = None
        
        curr_time = time.time()
        frame_time = curr_time - prev_time
        if frame_time > 0:
            fps = 1 / frame_time
            fps_smooth = fps_smooth * 0.9 + fps * 0.1
        prev_time = curr_time
        
        display_image = draw_ui(display_image, fps_smooth, current_mode, color_centers, synth_params)
        
        cv2.imshow('Hand Gesture Recognition', display_image)
        
        # 检查窗口是否被关闭（点击叉叉）
        if cv2.getWindowProperty('Hand Gesture Recognition', cv2.WND_PROP_VISIBLE) < 1:
            break
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('x'):
            break
        elif key == ord('1'):
            current_mode = MODE_MOUSE
        elif key == ord('2'):
            current_mode = MODE_GESTURE
        elif key == ord('3'):
            current_mode = MODE_ASCII
        elif key == ord('4'):
            current_mode = MODE_COLOR
            color_centers = None
        elif key == ord('5'):
            current_mode = MODE_SYNTH
        elif key == ord('r'):
            # 重置所有历史
            gesture_history.clear()
            hand_position_history.clear()
            hand_size_history.clear()
            print("History reset")
    
    print("Releasing resources...")
    # 停止合成器
    if synth is not None:
        synth.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
