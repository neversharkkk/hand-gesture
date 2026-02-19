from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout


class MainApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        label = Label(
            text='Hand Gesture Recognition\nVersion 1.4.0',
            font_size='24sp',
            halign='center'
        )
        layout.add_widget(label)
        
        btn = Button(
            text='Click Me',
            size_hint=(1, 0.3),
            font_size='20sp'
        )
        btn.bind(on_press=self.on_click)
        layout.add_widget(btn)
        
        return layout
    
    def on_click(self, instance):
        instance.text = 'Clicked!'


if __name__ == '__main__':
    MainApp().run()
