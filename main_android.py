import cv2
import numpy as np
import time
from collections import deque
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

try:
    from android_camera import get_camera_instance
except ImportError:
    def get_camera_instance(camera_index=0, width=640, height=480):
        class FallbackCamera:
            def __init__(self, camera_index, width, height):
                self.camera_index = camera_index
                self.width = width
                self.height = height
                self.cap = None
                self.permission_granted = True
            
            def open(self):
                self.cap = cv2.VideoCapture(self.camera_index)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    return True
                return False
            
            def read(self):
                if self.cap:
                    return self.cap.read()
                return False, None
            
            def release(self):
                if self.cap:
                    self.cap.release()
        
        return FallbackCamera(camera_index, width, height)
    Logger.info("Using fallback camera module")

MODE_GESTURE = 0
MODE_ASCII = 1
MODE_COLOR = 2
MODE_COLOR_16 = 3
current_mode = MODE_GESTURE

gesture_history = deque(maxlen=5)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60, detectShadows=False)
hand_position_history = deque(maxlen=3)
hand_size_history = deque(maxlen=3)
last_valid_contour = None
last_valid_time = 0

ASCII_CHARS = " .',:;ilI1|\\/()[]{}?_-+~=<>!@#$%&*#"
ASCII_COLORS = [
    (50, 50, 50), (70, 70, 70), (90, 90, 90), (110, 110, 110),
    (130, 130, 130), (150, 150, 150), (170, 170, 170), (190, 190, 190),
    (210, 210, 210), (230, 230, 230), (255, 255, 255), (255, 255, 150),
    (255, 230, 100), (255, 200, 50),
]

def smooth_gesture(gesture_name):
    if gesture_name:
        gesture_history.append(gesture_name)
    if len(gesture_history) >= 3:
        from collections import Counter
        counts = Counter(gesture_history)
        most_common = counts.most_common(1)[0]
        if most_common[1] >= 2:
            return most_common[0]
    return gesture_name if gesture_name else ""

def smooth_position(x, y, w, h):
    global hand_position_history, hand_size_history
    hand_position_history.append((x, y))
    hand_size_history.append((w, h))
    if len(hand_position_history) == 0:
        return x, y, w, h
    sum_x = sum(p[0] for p in hand_position_history)
    sum_y = sum(p[1] for p in hand_position_history)
    sum_w = sum(s[0] for s in hand_size_history)
    sum_h = sum(s[1] for s in hand_size_history)
    n = len(hand_position_history)
    return sum_x // n, sum_y // n, sum_w // n, sum_h // n

def detect_skin(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 15, 40], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lower_lab = np.array([0, 130, 130], dtype=np.uint8)
    upper_lab = np.array([255, 160, 160], dtype=np.uint8)
    mask_lab = cv2.inRange(lab, lower_lab, upper_lab)
    
    mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    mask = cv2.bitwise_or(mask, mask_lab)
    return mask

def process_hand(image):
    global last_valid_contour, last_valid_time
    h, w = image.shape[:2]
    skin_mask = detect_skin(image)
    fg_mask = bg_subtractor.apply(image, learningRate=0.001)
    fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
    combined = cv2.bitwise_and(skin_mask, fg_mask)
    if cv2.countNonZero(combined) < 1000:
        combined = skin_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        current_time = time.time()
        if last_valid_contour is not None and current_time - last_valid_time < 0.5:
            return last_valid_contour, None, combined
        return None, None, combined
    best_contour = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
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
        score = 0
        if 2000 < area < 80000:
            score += 30
        elif 1500 < area < 100000:
            score += 20
        if 0.2 < circularity < 0.7:
            score += 25
        elif 0.15 < circularity < 0.8:
            score += 15
        if 0.4 < solidity < 0.95:
            score += 25
        elif 0.3 < solidity < 0.98:
            score += 15
        if 0.5 < aspect_ratio < 2.0:
            score += 20
        elif 0.3 < aspect_ratio < 2.5:
            score += 10
        if len(hand_position_history) > 0:
            last_x, last_y = hand_position_history[-1]
            last_w, last_h = hand_size_history[-1]
            center_x = x + cw // 2
            center_y = y + ch // 2
            last_center_x = last_x + last_w // 2
            last_center_y = last_y + last_h // 2
            distance = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
            max_distance = np.sqrt(w**2 + h**2) * 0.4
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
    finger_count = 0
    if defects is not None and defects.shape[0] > 0:
        hull_points = cv2.convexHull(contour, returnPoints=True)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + cw // 2, y + ch // 2
        finger_tips = []
        for i, pt in enumerate(hull_points):
            pt = pt[0]
            if pt[1] < cy + ch * 0.1:
                dist = np.sqrt((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2)
                if dist > ch * 0.25:
                    finger_tips.append(pt)
        if len(finger_tips) > 0:
            filtered_tips = [finger_tips[0]]
            for tip in finger_tips[1:]:
                is_duplicate = False
                for ft in filtered_tips:
                    if np.sqrt((tip[0] - ft[0]) ** 2 + (tip[1] - ft[1]) ** 2) < cw * 0.2:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    filtered_tips.append(tip)
            finger_count = len(filtered_tips)
        valid_defects = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > 400:
                cnt_f = tuple(contour[f][0])
                if cnt_f[1] > cy - ch * 0.1:
                    valid_defects += 1
        if valid_defects > 0 and valid_defects < 5:
            finger_count = valid_defects + 1
        finger_count = min(max(finger_count, 0), 5)
    gesture_name = ""
    if solidity < 0.35:
        gesture_name = "Fist"
    elif solidity > 0.75 or finger_count >= 4:
        gesture_name = "Paper"
    elif finger_count == 2:
        gesture_name = "Victory"
    elif finger_count == 1:
        gesture_name = "One"
    elif finger_count == 3:
        gesture_name = "Three"
    else:
        gesture_name = "Point"
    gesture_name = smooth_gesture(gesture_name)
    return gesture_name, finger_count

def extract_8_dominant_colors(image):
    h, w = image.shape[:2]
    small = cv2.resize(image, (w // 12, h // 12))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3)
    if len(pixels) > 500:
        indices = np.random.choice(len(pixels), 500, replace=False)
        pixels = pixels[indices]
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 3.0)
    _, labels, centers = cv2.kmeans(pixels, 8, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    centers = np.uint8(centers)
    sorted_centers = centers[sorted_indices]
    return sorted_centers

def extract_16_dominant_colors(image):
    h, w = image.shape[:2]
    small = cv2.resize(image, (w // 8, h // 8))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3)
    if len(pixels) > 800:
        indices = np.random.choice(len(pixels), 800, replace=False)
        pixels = pixels[indices]
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 3.0)
    _, labels, centers = cv2.kmeans(pixels, 16, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    centers = np.uint8(centers)
    sorted_centers = centers[sorted_indices]
    return sorted_centers

def smooth_color_centers(new_centers, history, num_colors=8):
    if not history:
        return new_centers
    history_array = np.array(list(history), dtype=np.float32)
    new_array = np.array(new_centers, dtype=np.float32)
    smoothed = np.zeros_like(new_array)
    for i in range(num_colors):
        current = new_array[i]
        min_dist = float('inf')
        best_match = current.copy()
        for hist in history_array:
            for j in range(num_colors):
                dist = np.sum((hist[j] - current) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = hist[j]
        if min_dist < 8000:
            smoothed[i] = current * 0.5 + best_match * 0.5
        else:
            smoothed[i] = current
    return np.uint8(smoothed)

def create_color_palette(centers, num_colors=8):
    palette = []
    factors = [0.30, 0.42, 0.54, 0.66, 0.78, 0.90, 1.02, 1.15] if num_colors == 8 else [0.35, 0.50, 0.65, 0.80, 0.95, 1.10]
    for color in centers:
        r, g, b = color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        saturation = (max_val - min_val) / max_val if max_val > 0 else 0
        for factor in factors:
            if luminance < 80:
                adjusted_factor = factor * 1.15 + 0.1
            elif luminance < 150:
                adjusted_factor = factor * 1.05 + 0.05
            else:
                adjusted_factor = factor
            if saturation < 0.5:
                sat_boost = 1.3
            elif saturation < 0.7:
                sat_boost = 1.15
            else:
                sat_boost = 1.0
            new_r = int(min(255, max(0, r * adjusted_factor)))
            new_g = int(min(255, max(0, g * adjusted_factor)))
            new_b = int(min(255, max(0, b * adjusted_factor)))
            gray_val = 0.299 * new_r + 0.587 * new_g + 0.114 * new_b
            new_r = int(gray_val + (new_r - gray_val) * sat_boost)
            new_g = int(gray_val + (new_g - gray_val) * sat_boost)
            new_b = int(gray_val + (new_b - gray_val) * sat_boost)
            new_r = min(255, max(0, new_r))
            new_g = min(255, max(0, new_g))
            new_b = min(255, max(0, new_b))
            palette.append((new_b, new_g, new_r))
    return palette

def apply_color_mapping(image, centers, num_colors=8):
    h, w = image.shape[:2]
    palette = create_color_palette(centers, num_colors)
    palette = np.array(palette, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    centers_array = np.array(centers, dtype=np.float32)
    pixels = rgb.reshape(-1, 3)
    diff = pixels[:, np.newaxis, :] - centers_array[np.newaxis, :, :]
    distances = np.sum(diff ** 2, axis=2)
    color_indices = np.argmin(distances, axis=1)
    gray_flat = gray.reshape(-1).astype(np.float32)
    num_levels = 8 if num_colors == 8 else 6
    levels = (gray_flat / 255 * (num_levels - 1)).astype(np.int32)
    levels = np.clip(levels, 0, num_levels - 1)
    palette_indices = color_indices * num_levels + levels
    result_flat = palette[palette_indices]
    result = result_flat.reshape(h, w, 3)
    result = cv2.GaussianBlur(result, (3, 3), 0.5)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result

def create_ascii_art(image, contour, mask):
    if contour is None:
        return image
    h, w = image.shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    cw = min(w - x, cw + 2 * padding)
    ch = min(h - y, ch + 2 * padding)
    hand_region = image[y:y+ch, x:x+cw].copy()
    mask_region = mask[y:y+ch, x:x+cw]
    gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
    char_size = max(6, min(12, cw // 20))
    ascii_h = ch // char_size
    ascii_w = cw // char_size
    if ascii_h < 3 or ascii_w < 3:
        return image
    result = image.copy()
    for row in range(ascii_h):
        for col in range(ascii_w):
            y_start = row * char_size
            y_end = min((row + 1) * char_size, ch)
            x_start = col * char_size
            x_end = min((col + 1) * char_size, cw)
            block = gray[y_start:y_end, x_start:x_end]
            block_mask = mask_region[y_start:y_end, x_start:x_end]
            if cv2.countNonZero(block_mask) < (block_mask.size * 0.3):
                continue
            if block.size == 0:
                continue
            avg_brightness = np.mean(block)
            char_idx = int(avg_brightness / 255 * (len(ASCII_CHARS) - 1))
            char_idx = max(0, min(len(ASCII_CHARS) - 1, char_idx))
            color_idx = min(char_idx, len(ASCII_COLORS) - 1)
            char = ASCII_CHARS[char_idx]
            color = ASCII_COLORS[color_idx]
            draw_x = x + x_start + char_size // 2
            draw_y = y + y_start + char_size
            cv2.putText(result, char, (draw_x, draw_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, char_size / 15, 
                       color, 1, cv2.LINE_AA)
    return result


class HandGestureApp(App):
    def build(self):
        self.current_mode = MODE_GESTURE
        self.color_centers = None
        self.color_history = deque(maxlen=10)
        self.color_centers_16 = None
        self.color_history_16 = deque(maxlen=10)
        self.fps_smooth = 30
        self.prev_time = time.time()
        
        layout = BoxLayout(orientation='vertical')
        
        self.image_widget = Image(allow_stretch=True, keep_ratio=True)
        layout.add_widget(self.image_widget, size_hint=(1, 0.85))
        
        btn_layout = BoxLayout(size_hint=(1, 0.15), spacing=5, padding=5)
        
        self.btn_gesture = ToggleButton(text='手势', group='mode', state='down')
        self.btn_gesture.bind(on_press=lambda x: self.set_mode(MODE_GESTURE))
        btn_layout.add_widget(self.btn_gesture)
        
        self.btn_ascii = ToggleButton(text='ASCII', group='mode')
        self.btn_ascii.bind(on_press=lambda x: self.set_mode(MODE_ASCII))
        btn_layout.add_widget(self.btn_ascii)
        
        self.btn_color = ToggleButton(text='8色', group='mode')
        self.btn_color.bind(on_press=lambda x: self.set_mode(MODE_COLOR))
        btn_layout.add_widget(self.btn_color)
        
        self.btn_color_16 = ToggleButton(text='16色', group='mode')
        self.btn_color_16.bind(on_press=lambda x: self.set_mode(MODE_COLOR_16))
        btn_layout.add_widget(self.btn_color_16)
        
        btn_reset = Button(text='重置')
        btn_reset.bind(on_press=self.reset_history)
        btn_layout.add_widget(btn_reset)
        
        layout.add_widget(btn_layout)
        
        self.camera = get_camera_instance(camera_index=0, width=640, height=480)
        if not self.camera.open():
            Logger.error("HandGestureApp: Failed to open camera")
        
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)
        
        return layout
    
    def set_mode(self, mode):
        self.current_mode = mode
        if mode == MODE_COLOR:
            self.color_centers = None
        elif mode == MODE_COLOR_16:
            self.color_centers_16 = None
    
    def reset_history(self, instance):
        global gesture_history, hand_position_history, hand_size_history
        gesture_history.clear()
        hand_position_history.clear()
        hand_size_history.clear()
        Logger.info("History reset")
    
    def update_frame(self, dt):
        ret, frame = self.camera.read()
        if not ret:
            return
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        contour, defects, mask = process_hand(frame)
        gesture_name, finger_count = recognize_gesture(contour, defects)
        
        display_image = frame.copy()
        
        if contour is not None:
            if self.current_mode == MODE_GESTURE:
                cv2.drawContours(display_image, [contour], 0, (0, 255, 0), 2)
                x, y, cw, ch = cv2.boundingRect(contour)
                smooth_x, smooth_y, smooth_cw, smooth_ch = smooth_position(x, y, cw, ch)
                if gesture_name:
                    cv2.putText(display_image, gesture_name, (smooth_x, smooth_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if self.current_mode == MODE_ASCII and contour is not None:
            display_image = create_ascii_art(display_image, contour, mask)
        
        if self.current_mode == MODE_COLOR:
            new_colors = extract_8_dominant_colors(display_image)
            self.color_centers = smooth_color_centers(new_colors, self.color_history, 8)
            self.color_history.append(self.color_centers.copy())
            display_image = apply_color_mapping(display_image, self.color_centers, 8)
        
        if self.current_mode == MODE_COLOR_16:
            new_colors_16 = extract_16_dominant_colors(display_image)
            self.color_centers_16 = smooth_color_centers(new_colors_16, self.color_history_16, 16)
            self.color_history_16.append(self.color_centers_16.copy())
            display_image = apply_color_mapping(display_image, self.color_centers_16, 16)
        
        curr_time = time.time()
        frame_time = curr_time - self.prev_time
        if frame_time > 0:
            fps = 1 / frame_time
            self.fps_smooth = self.fps_smooth * 0.9 + fps * 0.1
        self.prev_time = curr_time
        
        cv2.putText(display_image, f'FPS:{int(self.fps_smooth)}', (10, 30), 
                   cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        
        mode_text = {MODE_GESTURE: 'Gesture', MODE_ASCII: 'ASCII', 
                     MODE_COLOR: '8-Color', MODE_COLOR_16: '16-Color'}
        cv2.putText(display_image, mode_text.get(self.current_mode, ''), (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        buf = cv2.flip(display_image, 0)
        buf = buf.tostring()
        texture = Texture.create(size=(display_image.shape[1], display_image.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture
    
    def on_stop(self):
        self.camera.release()


if __name__ == '__main__':
    HandGestureApp().run()
