from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.properties import StringProperty, NumericProperty
from kivy.graphics.texture import Texture
from kivy.metrics import dp, sp
import time
import threading

MODE_CAMERA = 0
MODE_GESTURE = 1
MODE_ASCII = 2
MODE_SYNTH = 3
MODE_SAMPLER = 4


class CameraHandler:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None
        self.frame = None
        self.frame_lock = threading.Lock()
        self.initialized = False
    
    def start(self):
        try:
            import cv2
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(3, 640)
                self.cap.set(4, 480)
                self.running = True
                self.thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.thread.start()
                self.initialized = True
                Logger.info('CameraHandler: Camera started')
                return True
        except Exception as e:
            Logger.error(f'CameraHandler: Error - {e}')
        return False
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.initialized = False
    
    def _capture_loop(self):
        import cv2
        while self.running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        with self.frame_lock:
                            self.frame = frame
            except Exception as e:
                Logger.error(f'CameraHandler: Capture error - {e}')
            time.sleep(0.03)
    
    def get_frame(self):
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
        return None


class HandGestureWidget(BoxLayout):
    fps_value = NumericProperty(0)
    status_text = StringProperty('Ready')
    gesture_text = StringProperty('None')
    
    def __init__(self, **kwargs):
        super(HandGestureWidget, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(5)
        self.padding = dp(5)
        
        self.current_mode = MODE_CAMERA
        self.camera_handler = CameraHandler()
        self.fps = 30
        self.prev_time = time.time()
        self.frame_count = 0
        self.camera_active = False
        
        self._build_ui()
        
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        Logger.info('HandGestureWidget: Initialized')
    
    def _build_ui(self):
        header = BoxLayout(size_hint=(1, None), height=dp(50), spacing=dp(5))
        
        self.title_label = Label(
            text='[b]Hand Gesture v1.6[/b]',
            markup=True,
            font_size=sp(18),
            halign='center'
        )
        header.add_widget(self.title_label)
        
        self.fps_label = Label(
            text='FPS: 0',
            font_size=sp(14),
            size_hint_x=0.3
        )
        header.add_widget(self.fps_label)
        
        self.add_widget(header)
        
        self.status_label = Label(
            text='Status: Ready',
            font_size=sp(14),
            size_hint=(1, None),
            height=dp(30),
            color=(0.5, 0.8, 0.5, 1)
        )
        self.add_widget(self.status_label)
        
        self.image_widget = Image(
            size_hint=(1, None),
            height=dp(350),
            allow_stretch=True,
            keep_ratio=True
        )
        self.add_widget(self.image_widget)
        
        self.info_panel = BoxLayout(
            size_hint=(1, None),
            height=dp(60),
            spacing=dp(5),
            padding=dp(5)
        )
        
        info_inner = BoxLayout(orientation='vertical', spacing=dp(2))
        
        self.gesture_label = Label(
            text='Gesture: None',
            font_size=sp(16),
            bold=True,
            color=(0.3, 0.7, 1, 1)
        )
        info_inner.add_widget(self.gesture_label)
        
        self.mode_label = Label(
            text='Mode: Camera',
            font_size=sp(14)
        )
        info_inner.add_widget(self.mode_label)
        
        self.info_panel.add_widget(info_inner)
        self.add_widget(self.info_panel)
        
        self.mode_panel = GridLayout(
            cols=5,
            size_hint=(1, None),
            height=dp(45),
            spacing=dp(3)
        )
        
        modes = [
            ('Camera', MODE_CAMERA),
            ('Gesture', MODE_GESTURE),
            ('ASCII', MODE_ASCII),
            ('Synth', MODE_SYNTH),
            ('TR-808', MODE_SAMPLER)
        ]
        
        self.mode_buttons = {}
        for text, mode in modes:
            btn = ToggleButton(
                text=text,
                font_size=sp(12),
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
            height=dp(50),
            spacing=dp(5)
        )
        
        self.btn_camera = Button(
            text='Start Camera',
            font_size=sp(14)
        )
        self.btn_camera.bind(on_press=self.toggle_camera)
        control_panel.add_widget(self.btn_camera)
        
        self.btn_reset = Button(
            text='Reset',
            font_size=sp(14),
            size_hint_x=0.4
        )
        self.btn_reset.bind(on_press=self.reset)
        control_panel.add_widget(self.btn_reset)
        
        self.add_widget(control_panel)
    
    def set_mode(self, mode):
        self.current_mode = mode
        mode_names = ['Camera', 'Gesture', 'ASCII', 'Synth', 'TR-808']
        self.mode_label.text = f'Mode: {mode_names[mode]}'
        self.status_label.text = f'Status: {mode_names[mode]} mode'
        
        for m, btn in self.mode_buttons.items():
            btn.state = 'down' if m == mode else 'normal'
        
        Logger.info(f'HandGestureWidget: Mode set to {mode}')
    
    def toggle_camera(self, instance):
        if self.camera_active:
            self.stop_camera()
            instance.text = 'Start Camera'
        else:
            self.start_camera()
            instance.text = 'Stop Camera'
    
    def start_camera(self):
        if self.camera_handler.start():
            self.camera_active = True
            self.status_label.text = 'Status: Camera active'
            Logger.info('HandGestureWidget: Camera started')
        else:
            self.status_label.text = 'Status: Camera error'
    
    def stop_camera(self):
        self.camera_handler.stop()
        self.camera_active = False
        self.status_label.text = 'Status: Camera stopped'
        self.image_widget.texture = None
    
    def reset(self, instance):
        self.stop_camera()
        self.current_mode = MODE_CAMERA
        self.gesture_label.text = 'Gesture: None'
        self.status_label.text = 'Status: Reset'
        
        for m, btn in self.mode_buttons.items():
            btn.state = 'down' if m == MODE_CAMERA else 'normal'
    
    def update(self, dt):
        curr_time = time.time()
        frame_time = curr_time - self.prev_time
        if frame_time > 0:
            self.fps = self.fps * 0.9 + (1.0 / frame_time) * 0.1
        self.prev_time = curr_time
        
        self.frame_count += 1
        if self.frame_count % 5 == 0:
            self.fps_label.text = f'FPS: {int(self.fps)}'
        
        if self.camera_active:
            frame = self.camera_handler.get_frame()
            if frame is not None:
                self._display_frame(frame)
    
    def _display_frame(self, frame):
        try:
            import cv2
            frame = cv2.flip(frame, 0)
            buf = frame.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgba')
            texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
            self.image_widget.texture = texture
        except Exception as e:
            Logger.error(f'HandGestureWidget: Display error - {e}')


class HandGestureApp(App):
    def build(self):
        Logger.info('HandGestureApp: Building app...')
        return HandGestureWidget()
    
    def on_stop(self):
        Logger.info('HandGestureApp: Stopping...')


if __name__ == '__main__':
    HandGestureApp().run()
