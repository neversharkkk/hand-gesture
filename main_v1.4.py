from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.properties import StringProperty
import time


MODE_CAMERA = 0
MODE_GESTURE = 1
MODE_ASCII = 2


class MainWidget(BoxLayout):
    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = 5
        self.padding = 5
        
        self.current_mode = MODE_CAMERA
        self.camera_index = 0
        self.camera = None
        self.camera_active = False
        self.fps = 30
        self.prev_time = time.time()
        
        self.title_label = Label(
            text='Hand Gesture Recognition v1.4',
            font_size='20sp',
            size_hint=(1, 0.08),
            bold=True
        )
        self.add_widget(self.title_label)
        
        self.status_label = Label(
            text='Status: Ready',
            font_size='14sp',
            size_hint=(1, 0.06)
        )
        self.add_widget(self.status_label)
        
        self.camera_container = BoxLayout(size_hint=(1, 0.72))
        self.add_widget(self.camera_container)
        
        self.info_label = Label(
            text='FPS: 0',
            font_size='12sp',
            size_hint=(1, 0.06)
        )
        self.add_widget(self.info_label)
        
        btn_layout = GridLayout(cols=5, size_hint=(1, 0.08), spacing=3)
        
        self.btn_camera = Button(text='Camera')
        self.btn_camera.bind(on_press=self.toggle_camera)
        btn_layout.add_widget(self.btn_camera)
        
        self.btn_switch = Button(text='Switch')
        self.btn_switch.bind(on_press=self.switch_camera)
        btn_layout.add_widget(self.btn_switch)
        
        self.btn_gesture = ToggleButton(text='Gesture', group='mode')
        self.btn_gesture.bind(on_press=lambda x: self.set_mode(MODE_GESTURE))
        btn_layout.add_widget(self.btn_gesture)
        
        self.btn_ascii = ToggleButton(text='ASCII', group='mode')
        self.btn_ascii.bind(on_press=lambda x: self.set_mode(MODE_ASCII))
        btn_layout.add_widget(self.btn_ascii)
        
        self.btn_reset = Button(text='Reset')
        self.btn_reset.bind(on_press=self.reset)
        btn_layout.add_widget(self.btn_reset)
        
        self.add_widget(btn_layout)
        
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        
        Logger.info('MainWidget: Initialized')
    
    def set_mode(self, mode):
        self.current_mode = mode
        self.status_label.text = f'Mode: {["Camera", "Gesture", "ASCII"][mode]}'
        Logger.info(f'MainWidget: Mode set to {mode}')
    
    def toggle_camera(self, instance):
        if self.camera_active:
            self.stop_camera()
            instance.text = 'Camera'
        else:
            self.start_camera()
            instance.text = 'Stop'
    
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
            Logger.info(f'MainWidget: Camera {self.camera_index} started')
        except Exception as e:
            Logger.error(f'MainWidget: Camera error - {e}')
            self.status_label.text = 'Camera error'
    
    def stop_camera(self):
        if self.camera:
            self.camera.play = False
            self.camera_container.clear_widgets()
            self.camera = None
        self.camera_active = False
        self.status_label.text = 'Camera stopped'
        Logger.info('MainWidget: Camera stopped')
    
    def switch_camera(self, instance):
        self.stop_camera()
        self.camera_index = 1 - self.camera_index
        self.start_camera()
    
    def reset(self, instance):
        self.stop_camera()
        self.camera_index = 0
        self.current_mode = MODE_CAMERA
        self.status_label.text = 'Reset'
    
    def update(self, dt):
        curr_time = time.time()
        frame_time = curr_time - self.prev_time
        if frame_time > 0:
            self.fps = self.fps * 0.9 + (1.0 / frame_time) * 0.1
        self.prev_time = curr_time
        
        if self.camera_active:
            self.info_label.text = f'FPS: {int(self.fps)} | Mode: {["Camera", "Gesture", "ASCII"][self.current_mode]}'


class HandGestureApp(App):
    def build(self):
        Logger.info('HandGestureApp: Building app...')
        return MainWidget()
    
    def on_stop(self):
        Logger.info('HandGestureApp: Stopping...')


if __name__ == '__main__':
    HandGestureApp().run()
