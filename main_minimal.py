import numpy as np
import time
import threading
from collections import deque
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.properties import StringProperty

MODE_CAMERA = 0
MODE_GESTURE = 1
MODE_ASCII = 2

ASCII_CHARS = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"


class SimpleCamera:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None
        self.frame = None
        self.frame_lock = threading.Lock()
        
    def start(self):
        try:
            import cv2
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.running = True
                self.thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.thread.start()
                Logger.info("Camera started")
                return True
        except Exception as e:
            Logger.error(f"Camera init error: {e}")
        return False
    
    def stop(self):
        self.running = False
        if self.thread:
            try:
                self.thread.join(timeout=1.0)
            except:
                pass
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
    
    def _capture_loop(self):
        import cv2
        while self.running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        with self.frame_lock:
                            self.frame = frame.copy()
            except Exception as e:
                Logger.error(f"Capture error: {e}")
            time.sleep(0.02)
    
    def get_frame(self):
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
        return None


class SimpleHandDetector:
    def __init__(self):
        self.last_valid_contour = None
        self.last_valid_time = 0
        self.gesture_history = deque(maxlen=5)
        self.bg_subtractor = None
        self.initialized = False
        
    def initialize(self):
        try:
            import cv2
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60, detectShadows=False)
            self.initialized = True
            Logger.info("Detector initialized")
        except Exception as e:
            Logger.error(f"Detector init error: {e}")
    
    def detect(self, frame):
        import cv2
        if not self.initialized:
            return None, None, 0
        
        try:
            h, w = frame.shape[:2]
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
            upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_contour = None
            max_area = 0
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 2000 < area < 150000:
                    if area > max_area:
                        max_area = area
                        best_contour = cnt
            
            if best_contour is not None:
                self.last_valid_contour = best_contour
                self.last_valid_time = time.time()
            elif self.last_valid_contour is not None and (time.time() - self.last_valid_time) < 0.5:
                best_contour = self.last_valid_contour
            
            return best_contour, mask, max_area
        except Exception as e:
            Logger.error(f"Detect error: {e}")
            return None, None, 0
    
    def recognize_gesture(self, frame, contour):
        import cv2
        if contour is None:
            return 'None', 0
        
        try:
            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)
            rect_w, rect_h = rect[1]
            solidity = area / (rect_w * rect_h) if (rect_w * rect_h) > 0 else 0
            
            gesture = 'Paper'
            if solidity > 0.8:
                gesture = 'Fist'
            
            self.gesture_history.append(gesture)
            if len(self.gesture_history) >= 3:
                from collections import Counter
                counter = Counter(self.gesture_history)
                gesture = counter.most_common(1)[0][0]
            
            return gesture, 0
        except Exception as e:
            Logger.error(f"Recognize error: {e}")
            return 'Unknown', 0


def create_simple_ascii(image, contour):
    import cv2
    if contour is None:
        return image
    
    try:
        h, w = image.shape[:2]
        x, y, cw, ch = cv2.boundingRect(contour)
        cw = min(cw, w - x)
        ch = min(ch, h - y)
        
        if cw < 10 or ch < 10:
            return image
        
        roi = image[y:y+ch, x:x+cw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        char_size = max(6, min(10, cw // 25))
        ascii_h = ch // char_size
        ascii_w = cw // char_size
        
        if ascii_h < 3 or ascii_w < 3:
            return image
        
        result = image.copy()
        
        for row in range(ascii_h):
            for col in range(ascii_w):
                if np.random.random() < 0.3:
                    continue
                
                y_start = row * char_size
                y_end = min((row + 1) * char_size, ch)
                x_start = col * char_size
                x_end = min((col + 1) * char_size, cw)
                
                block = gray[y_start:y_end, x_start:x_end]
                if block.size == 0:
                    continue
                
                avg_brightness = np.mean(block)
                char_idx = int(avg_brightness / 255 * (len(ASCII_CHARS) - 1))
                char = ASCII_CHARS[max(0, min(len(ASCII_CHARS) - 1, char_idx))]
                
                blue_val = int((50 + avg_brightness * 0.4) * 0.7)
                green_val = int((150 + avg_brightness * 0.3) * 0.7)
                red_val = int((100 + avg_brightness * 0.2) * 0.7)
                
                draw_x = x + x_start + char_size // 2
                draw_y = y + y_start + char_size
                cv2.putText(result, char, (draw_x, draw_y),
                           cv2.FONT_HERSHEY_SIMPLEX, char_size / 20,
                           (blue_val, green_val, red_val), 1, cv2.LINE_AA)
        
        return result
    except Exception as e:
        Logger.error(f"ASCII error: {e}")
        return image


class MinimalHandGestureApp(App):
    fps_text = StringProperty('FPS: 0')
    status_text = StringProperty('Initializing...')
    
    def build(self):
        self.current_mode = MODE_CAMERA
        self.camera = SimpleCamera()
        self.detector = SimpleHandDetector()
        self.fps = 30
        self.prev_time = time.time()
        self.frame_count = 0
        self.ready = False
        
        layout = BoxLayout(orientation='vertical')
        
        self.display = Image(allow_stretch=True, keep_ratio=True)
        layout.add_widget(self.display, size_hint=(1, 0.85))
        
        btn_layout = GridLayout(cols=4, size_hint=(1, 0.15), spacing=5, padding=5)
        
        modes = [('Camera', MODE_CAMERA), ('Gesture', MODE_GESTURE), ('ASCII', MODE_ASCII)]
        
        self.buttons = {}
        for name, mode in modes:
            btn = ToggleButton(text=name, group='mode')
            if mode == MODE_CAMERA:
                btn.state = 'down'
            btn.bind(on_press=lambda x, m=mode: self.set_mode(m))
            btn_layout.add_widget(btn)
            self.buttons[mode] = btn
        
        btn_reset = Button(text='Reset')
        btn_reset.bind(on_press=self.reset)
        btn_layout.add_widget(btn_reset)
        
        layout.add_widget(btn_layout)
        
        Clock.schedule_once(self._init, 1.0)
        Clock.schedule_interval(self.update_loop, 1.0 / 25.0)
        
        Logger.info("App built")
        return layout
    
    def _init(self, dt):
        Logger.info("Initializing camera...")
        self.status_text = "Starting camera..."
        if self.camera.start():
            Logger.info("Camera OK")
            self.status_text = "Initializing detector..."
            self.detector.initialize()
            self.ready = True
            self.status_text = "Ready"
            Logger.info("All initialized")
        else:
            self.status_text = "Camera failed"
            Logger.error("Camera start failed")
    
    def set_mode(self, mode):
        self.current_mode = mode
        for m, btn in self.buttons.items():
            if m == mode:
                btn.state = 'down'
            else:
                btn.state = 'normal'
    
    def reset(self, instance):
        self.detector.last_valid_contour = None
    
    def update_loop(self, dt):
        try:
            curr_time = time.time()
            frame_time = curr_time - self.prev_time
            if frame_time > 0:
                self.fps = self.fps * 0.9 + (1.0 / frame_time) * 0.1
            self.prev_time = curr_time
            
            if not self.ready:
                display = np.zeros((480, 640, 3), dtype=np.uint8)
                self._draw_text(display, self.status_text, (320, 240), 1.0, (255,255,255))
            else:
                frame = self.camera.get_frame()
                if frame is None:
                    display = np.zeros((480, 640, 3), dtype=np.uint8)
                    self._draw_text(display, 'No signal', (320, 240), 1.0, (255,255,255))
                else:
                    display = frame.copy()
                    
                    contour, mask, area = self.detector.detect(display)
                    gesture, _ = self.detector.recognize_gesture(display, contour)
                    
                    if contour is not None:
                        import cv2
                        if self.current_mode in [MODE_GESTURE, MODE_ASCII]:
                            cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
                        
                        if self.current_mode == MODE_ASCII:
                            display = create_simple_ascii(display, contour)
                    
                    self._draw_ui(display, gesture)
            
            import cv2
            buf = cv2.flip(display, 0).tobytes()
            texture = Texture.create(size=(display.shape[1], display.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.display.texture = texture
            
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                self.fps_text = f'FPS: {int(self.fps)}'
            
        except Exception as e:
            Logger.error(f"Update error: {e}")
            import traceback
            Logger.error(traceback.format_exc())
    
    def _draw_text(self, img, text, center, scale, color):
        import cv2
        h, w = img.shape[:2]
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        x = int(center[0] - textsize[0] / 2)
        y = int(center[1] + textsize[1] / 2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    
    def _draw_ui(self, display, gesture):
        import cv2
        h, w = display.shape[:2]
        
        mode_names = ['Camera', 'Gesture', 'ASCII']
        
        cv2.putText(display, f'Mode: {mode_names[self.current_mode]}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, f'FPS: {int(self.fps)}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if self.current_mode in [MODE_GESTURE, MODE_ASCII]:
            cv2.putText(display, f'Gesture: {gesture}', (10, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def on_stop(self):
        Logger.info("App stopping")
        self.camera.stop()


if __name__ == '__main__':
    try:
        Logger.info("Starting app...")
        MinimalHandGestureApp().run()
    except Exception as e:
        Logger.error(f"App crash: {e}")
        import traceback
        Logger.error(traceback.format_exc())
