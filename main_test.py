from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.properties import StringProperty
import time


class TestApp(App):
    status_text = StringProperty('Ready')
    fps_text = StringProperty('FPS: 0')
    
    def build(self):
        Logger.info("Building test app...")
        self.fps = 30
        self.prev_time = time.time()
        self.count = 0
        
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        self.label = Label(
            text='Hand Gesture Recognition\nTest Version',
            font_size='24sp',
            halign='center'
        )
        layout.add_widget(self.label)
        
        self.status_label = Label(
            text='Status: Ready',
            font_size='18sp'
        )
        layout.add_widget(self.status_label)
        
        self.fps_label = Label(
            text='FPS: 0',
            font_size='16sp'
        )
        layout.add_widget(self.fps_label)
        
        btn = Button(
            text='Click Me!',
            size_hint=(1, 0.3),
            font_size='20sp'
        )
        btn.bind(on_press=self.on_button_press)
        layout.add_widget(btn)
        
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        
        Logger.info("Test app built successfully")
        return layout
    
    def on_button_press(self, instance):
        self.count += 1
        self.status_text = f'Clicked {self.count} times'
        self.status_label.text = f'Status: {self.status_text}'
        Logger.info(f"Button clicked: {self.count}")
    
    def update(self, dt):
        curr_time = time.time()
        frame_time = curr_time - self.prev_time
        if frame_time > 0:
            self.fps = self.fps * 0.9 + (1.0 / frame_time) * 0.1
        self.prev_time = curr_time
        self.fps_label.text = f'FPS: {int(self.fps)}'


if __name__ == '__main__':
    Logger.info("Starting TestApp...")
    TestApp().run()
