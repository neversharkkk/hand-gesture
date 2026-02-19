import kivy
kivy.require('2.2.1')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.logger import Logger


class MainWidget(BoxLayout):
    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = 10
        self.padding = 10
        
        self.title_label = Label(
            text='Hand Gesture Recognition',
            font_size='24sp',
            size_hint=(1, 0.1)
        )
        self.add_widget(self.title_label)
        
        self.status_label = Label(
            text='Status: Ready',
            font_size='16sp',
            size_hint=(1, 0.1)
        )
        self.add_widget(self.status_label)
        
        self.camera = None
        self.camera_container = BoxLayout(size_hint=(1, 0.6))
        self.add_widget(self.camera_container)
        
        btn_layout = BoxLayout(size_hint=(1, 0.1), spacing=5)
        
        self.btn_camera = Button(text='Start Camera')
        self.btn_camera.bind(on_press=self.toggle_camera)
        btn_layout.add_widget(self.btn_camera)
        
        self.btn_switch = Button(text='Switch')
        self.btn_switch.bind(on_press=self.switch_camera)
        btn_layout.add_widget(self.btn_switch)
        
        self.add_widget(btn_layout)
        
        self.camera_index = 0
        self.camera_active = False
        
        Clock.schedule_once(self.init_camera, 1.0)
    
    def init_camera(self, dt):
        Logger.info('MainWidget: Initializing...')
        self.status_label.text = 'Status: Initialized'
    
    def toggle_camera(self, instance):
        if self.camera_active:
            self.stop_camera()
            self.btn_camera.text = 'Start Camera'
        else:
            self.start_camera()
            self.btn_camera.text = 'Stop Camera'
    
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
            self.status_label.text = f'Status: Camera {self.camera_index} active'
            Logger.info(f'MainWidget: Camera {self.camera_index} started')
        except Exception as e:
            Logger.error(f'MainWidget: Camera error - {e}')
            self.status_label.text = f'Status: Camera error'
    
    def stop_camera(self):
        if self.camera:
            self.camera.play = False
            self.camera_container.clear_widgets()
            self.camera = None
        self.camera_active = False
        self.status_label.text = 'Status: Camera stopped'
        Logger.info('MainWidget: Camera stopped')
    
    def switch_camera(self, instance):
        self.stop_camera()
        self.camera_index = 1 - self.camera_index
        self.start_camera()


class HandGestureApp(App):
    def build(self):
        Logger.info('HandGestureApp: Building app...')
        return MainWidget()
    
    def on_stop(self):
        Logger.info('HandGestureApp: Stopping...')


if __name__ == '__main__':
    HandGestureApp().run()
