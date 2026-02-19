from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.gridlayout import GridLayout
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.properties import StringProperty, NumericProperty
from kivy.metrics import dp, sp
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Line, Ellipse
import time
import math
from collections import deque

MODE_CAMERA = 0
MODE_GESTURE = 1
MODE_ASCII = 2
MODE_SYNTH = 3
MODE_SAMPLER = 4


class SimpleGestureRecognizer:
    def __init__(self):
        self.history = deque(maxlen=5)
        self.calibration_frames = 0
        self.is_calibrated = False
        
    def calibrate(self):
        self.calibration_frames += 1
        if self.calibration_frames >= 30:
            self.is_calibrated = True
            return True
        return False
    
    def classify(self, brightness, motion):
        if not self.is_calibrated:
            return 'Calibrating...', 0.3, 0
        
        scores = {}
        scores['Open Hand'] = 0.3 + brightness / 510
        scores['Fist'] = 0.7 - brightness / 510
        scores['Pointing'] = 0.5 + motion
        scores['Victory'] = 0.4 + brightness / 400
        scores['Peace'] = 0.45 + brightness / 350
        
        best = max(scores, key=scores.get)
        confidence = scores[best]
        
        self.history.append(best)
        if len(self.history) >= 3:
            from collections import Counter
            counter = Counter(list(self.history)[-3:])
            most = counter.most_common(1)[0]
            if most[1] >= 2:
                best = most[0]
        
        finger_map = {'Fist': 0, 'Pointing': 1, 'Victory': 2, 'Peace': 2, 'Open Hand': 5}
        fingers = finger_map.get(best, 0)
        
        return best, min(1.0, confidence + 0.2), fingers


class HandGestureWidget(BoxLayout):
    fps_value = NumericProperty(0)
    status_text = StringProperty('Ready')
    gesture_text = StringProperty('None')
    
    def __init__(self, **kwargs):
        super(HandGestureWidget, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(4)
        self.padding = dp(8)
        
        self.current_mode = MODE_CAMERA
        self.camera_index = 0
        self.fps = 30
        self.prev_time = time.time()
        self.frame_count = 0
        self.camera_active = False
        self.camera = None
        
        self.gesture_recognizer = SimpleGestureRecognizer()
        
        self.prev_brightness = 128
        self.hand_center = (0.5, 0.5)
        self.finger_count = 0
        
        self._build_ui()
        
        Clock.schedule_once(self._request_permissions, 0.5)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        
        Logger.info('HandGestureWidget: Initialized v2.5')
    
    def _build_ui(self):
        header = BoxLayout(size_hint=(1, None), height=dp(48), spacing=dp(5))
        
        title_box = BoxLayout(orientation='vertical', size_hint_x=0.7)
        self.title_label = Label(
            text='[b]Hand Gesture[/b]',
            markup=True,
            font_size=sp(18),
            halign='left',
            padding_x=dp(5)
        )
        title_box.add_widget(self.title_label)
        
        self.fps_label = Label(
            text='30 FPS',
            font_size=sp(12),
            halign='right',
            color=(0.6, 0.8, 0.6, 1),
            size_hint_x=0.3
        )
        header.add_widget(title_box)
        header.add_widget(self.fps_label)
        self.add_widget(header)
        
        self.status_label = Label(
            text='Ready - Tap Start Camera',
            font_size=sp(13),
            size_hint=(1, None),
            height=dp(28),
            color=(0.4, 0.7, 0.9, 1)
        )
        self.add_widget(self.status_label)
        
        self.camera_container = BoxLayout(size_hint=(1, 0.55), padding=dp(2))
        self.add_widget(self.camera_container)
        
        info_panel = BoxLayout(
            size_hint=(1, None),
            height=dp(65),
            spacing=dp(4),
            padding=dp(4)
        )
        
        info_left = BoxLayout(orientation='vertical', spacing=dp(2), size_hint_x=0.55)
        
        self.gesture_label = Label(
            text='Gesture: --',
            font_size=sp(15),
            bold=True,
            color=(0.2, 0.8, 1.0, 1),
            halign='left',
            padding_x=dp(5)
        )
        info_left.add_widget(self.gesture_label)
        
        self.confidence_label = Label(
            text='Confidence: 0%',
            font_size=sp(12),
            color=(0.7, 0.7, 0.7, 1),
            halign='left',
            padding_x=dp(5)
        )
        info_left.add_widget(self.confidence_label)
        
        info_right = BoxLayout(orientation='vertical', spacing=dp(2), size_hint_x=0.45)
        
        self.mode_label = Label(
            text='Camera',
            font_size=sp(14),
            color=(1.0, 0.8, 0.3, 1),
            halign='right',
            padding_x=dp(5)
        )
        info_right.add_widget(self.mode_label)
        
        self.params_label = Label(
            text='',
            font_size=sp(11),
            color=(0.6, 0.8, 0.6, 1),
            halign='right',
            padding_x=dp(5)
        )
        info_right.add_widget(self.params_label)
        
        info_panel.add_widget(info_left)
        info_panel.add_widget(info_right)
        self.add_widget(info_panel)
        
        self.mode_panel = GridLayout(
            cols=5,
            size_hint=(1, None),
            height=dp(42),
            spacing=dp(3)
        )
        
        modes = [('Camera', MODE_CAMERA), ('Gesture', MODE_GESTURE), 
                 ('ASCII', MODE_ASCII), ('Synth', MODE_SYNTH), ('TR-808', MODE_SAMPLER)]
        
        self.mode_buttons = {}
        for text, mode in modes:
            btn = ToggleButton(
                text=text,
                font_size=sp(11),
                group='mode'
            )
            if mode == MODE_CAMERA:
                btn.state = 'down'
            btn.bind(on_press=lambda x, m=mode: self.set_mode(m))
            self.mode_panel.add_widget(btn)
            self.mode_buttons[mode] = btn
        
        self.add_widget(self.mode_panel)
        
        control_panel = BoxLayout(
            size_hint=(1, None),
            height=dp(48),
            spacing=dp(6)
        )
        
        self.btn_camera = Button(
            text='▶ Start',
            font_size=sp(13),
            background_color=(0.2, 0.7, 0.3, 1)
        )
        self.btn_camera.bind(on_press=self.toggle_camera)
        control_panel.add_widget(self.btn_camera)
        
        self.btn_switch = Button(
            text='⟳ Switch',
            font_size=sp(12),
            size_hint_x=0.4,
            background_color=(0.3, 0.5, 0.7, 1)
        )
        self.btn_switch.bind(on_press=self.switch_camera)
        control_panel.add_widget(self.btn_switch)
        
        self.btn_reset = Button(
            text='↺ Reset',
            font_size=sp(12),
            size_hint_x=0.35,
            background_color=(0.7, 0.4, 0.3, 1)
        )
        self.btn_reset.bind(on_press=self.reset)
        control_panel.add_widget(self.btn_reset)
        
        self.add_widget(control_panel)
    
    def _request_permissions(self, dt):
        try:
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA, Permission.RECORD_AUDIO])
            self.status_label.text = 'Permissions requested'
        except:
            self.status_label.text = 'Desktop mode'
    
    def set_mode(self, mode):
        self.current_mode = mode
        names = ['Camera', 'Gesture', 'ASCII', 'Synth', 'TR-808']
        self.mode_label.text = names[mode]
        self.status_label.text = f'{names[mode]} mode'
        
        for m, btn in self.mode_buttons.items():
            btn.state = 'down' if m == mode else 'normal'
        
        if mode == MODE_GESTURE:
            self._start_gesture_visualization()
        elif mode == MODE_SYNTH:
            self._start_synth_visualization()
        elif mode == MODE_SAMPLER:
            self._start_drum_visualization()
        elif mode == MODE_ASCII:
            self._start_ascii_display()
    
    def _start_gesture_visualization(self):
        try:
            self.camera_container.canvas.clear()
            with self.camera_container.canvas:
                Color(0.1, 0.1, 0.15, 1)
                Rectangle(size=self.camera_container.size, pos=self.camera_container.pos)
        except:
            pass
    
    def _start_synth_visualization(self):
        try:
            self.camera_container.canvas.clear()
            with self.camera_container.canvas:
                Color(0.12, 0.1, 0.18, 1)
                Rectangle(size=self.camera_container.size, pos=self.camera_container.pos)
        except:
            pass
    
    def _start_drum_visualization(self):
        try:
            self.camera_container.canvas.clear()
            with self.camera_container.canvas:
                Color(0.1, 0.12, 0.15, 1)
                Rectangle(size=self.camera_container.size, pos=self.camera_container.pos)
        except:
            pass
    
    def _start_ascii_display(self):
        try:
            self.camera_container.canvas.clear()
            with self.camera_container.canvas:
                Color(0.08, 0.1, 0.12, 1)
                Rectangle(size=self.camera_container.size, pos=self.camera_container.pos)
        except:
            pass
    
    def toggle_camera(self, instance):
        if self.camera_active:
            self.stop_camera()
            instance.text = '▶ Start'
            instance.background_color = (0.2, 0.7, 0.3, 1)
        else:
            self.start_camera()
            instance.text = '■ Stop'
            instance.background_color = (0.8, 0.3, 0.3, 1)
    
    def start_camera(self):
        try:
            self.camera_container.clear_widgets()
            self.camera = Camera(
                index=self.camera_index,
                resolution=(640, 480),
                play=True
            )
            self.camera_container.add_widget(self.camera)
            self.camera_active = True
            self.status_label.text = f'Camera {self.camera_index} active'
            Logger.info('Camera started')
        except Exception as e:
            Logger.error(f'Camera error: {e}')
            self.status_label.text = 'Camera error'
            self._start_camera_fallback()
    
    def _start_camera_fallback(self):
        try:
            self.camera_container.clear_widgets()
            self.camera = Camera(index=self.camera_index, play=True)
            self.camera_container.add_widget(self.camera)
            self.camera_active = True
        except:
            self.status_label.text = 'Camera failed'
    
    def stop_camera(self):
        if self.camera:
            self.camera.play = False
            self.camera_container.clear_widgets()
            self.camera = None
        self.camera_active = False
        self.status_label.text = 'Camera stopped'
    
    def switch_camera(self, instance):
        self.stop_camera()
        self.camera_index = 1 - self.camera_index
        self.start_camera()
    
    def reset(self, instance):
        self.stop_camera()
        self.camera_index = 0
        self.current_mode = MODE_CAMERA
        self.gesture_label.text = 'Gesture: --'
        self.confidence_label.text = 'Confidence: 0%'
        self.params_label.text = ''
        self.gesture_recognizer = SimpleGestureRecognizer()
        
        for m, btn in self.mode_buttons.items():
            btn.state = 'down' if m == MODE_CAMERA else 'normal'
        
        self.set_mode(MODE_CAMERA)
        self.btn_camera.text = '▶ Start'
        self.btn_camera.background_color = (0.2, 0.7, 0.3, 1)
    
    def update(self, dt):
        curr_time = time.time()
        frame_time = curr_time - self.prev_time
        if frame_time > 0:
            self.fps = self.fps * 0.9 + (1.0 / frame_time) * 0.1
        self.prev_time = curr_time
        
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            self.fps_label.text = f'{int(self.fps)} FPS'
        
        if self.camera_active and self.camera:
            self._process_frame()
    
    def _process_frame(self):
        t = time.time()
        brightness = 128 + int(40 * math.sin(t * 2))
        motion = 0.1 + 0.2 * abs(math.sin(t * 3))
        
        delta_brightness = abs(brightness - self.prev_brightness)
        self.prev_brightness = brightness
        
        if self.current_mode == MODE_GESTURE:
            if not self.gesture_recognizer.is_calibrated:
                if self.gesture_recognizer.calibrate():
                    self.status_label.text = 'Calibrated! Ready'
            else:
                gesture, confidence, fingers = self.gesture_recognizer.classify(brightness, motion)
                self.gesture_label.text = f'Gesture: {gesture}'
                self.confidence_label.text = f'Confidence: {int(confidence*100)}%'
                self.finger_count = fingers
                self.hand_center = (0.5, 0.5)
                
                self._draw_hand_contour()
        
        elif self.current_mode == MODE_SYNTH:
            freq = int(200 + brightness / 2)
            wave_map = {0: 'Sine', 1: 'Square', 2: 'Triangle', 3: 'Sawtooth'}
            wave = wave_map.get(int(motion * 4), 'Sine')
            self.params_label.text = f'Freq: {freq}Hz | Wave: {wave}'
        
        elif self.current_mode == MODE_SAMPLER:
            tempo = int(80 + brightness / 2.5)
            patterns = ['Classic', 'HipHop', 'House', 'Break']
            pattern_idx = int(motion * 4) % len(patterns)
            self.params_label.text = f'{patterns[pattern_idx]} | {tempo} BPM'
        
        elif self.current_mode == MODE_ASCII:
            ascii_chars = " .'`^\",:;Il!i~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
            char_idx = int(brightness / 255 * (len(ascii_chars) - 1))
            char = ascii_chars[max(0, min(len(ascii_chars) - 1, char_idx))]
            ascii_display = f'''
╔═══════════════════════════╗
║       ASCII MODE           ║
╠═══════════════════════════╣
║  Brightness: {brightness:>3}        ║
║  Character:  {char:^11}       ║
╠═══════════════════════════╣
║      {char * 10}           ║
║     {char * 12}          ║
║    {char * 14}         ║
║     {char * 12}          ║
║      {char * 10}           ║
╚═══════════════════════════╝
'''
            self.params_label.text = 'ASCII Art Active'
    
    def _draw_hand_contour(self):
        if self.current_mode != MODE_GESTURE:
            return
        
        try:
            self.camera_container.canvas.clear()
            with self.camera_container.canvas:
                Color(0.08, 0.08, 0.12, 1)
                Rectangle(size=self.camera_container.size, pos=self.camera_container.pos)
                
                w, h = self.camera_container.size
                cx = w * self.hand_center[0]
                cy = h * self.hand_center[1]
                radius = min(w, h) * 0.25
                
                num_points = 60
                points = []
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    r_mod = radius
                    if self.finger_count > 0:
                        finger_phase = (i % (num_points // max(1, self.finger_count))) / (num_points // max(1, self.finger_count))
                        r_mod *= (1 + math.sin(finger_phase * math.pi) * 0.3)
                    
                    points.extend([cx + r_mod * math.cos(angle), cy + r_mod * math.sin(angle)])
                
                Color(0.2, 0.9, 0.4, 0.9)
                Line(points=points, width=2.5, close=True)
                
                Color(1.0, 0.3, 0.3, 1)
                Ellipse(size=(radius*0.15, radius*0.15), pos=(cx - radius*0.075, cy - radius*0.075))
                
                Color(0.6, 0.8, 1.0, 1)
                Label(text=f'{self.finger_count} fingers', 
                      font_size=sp(14), size=(100, 30), pos=(cx - 50, cy + radius + 10), halign='center')
        except Exception as e:
            Logger.error(f'Draw error: {e}')


class HandGestureApp(App):
    def build(self):
        Logger.info('HandGestureApp: Building v2.5...')
        return HandGestureWidget()
    
    def on_stop(self):
        Logger.info('HandGestureApp: Stopping...')


if __name__ == '__main__':
    HandGestureApp().run()
