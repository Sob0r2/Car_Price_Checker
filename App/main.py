from dotenv import load_dotenv
from Final_Screen.finalscreen import FinalScreen
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.screenmanager import MDScreenManager
from Main_Screen.mainscreen import MainScreen

load_dotenv()

Builder.load_file("Main_Screen/mainscreen.kv")
Builder.load_file("Final_Screen/finalscreen.kv")

LabelBase.register(name="TITLE_FONT", fn_regular="Graphics/Asutenan!.ttf")
LabelBase.register(name="TEXT_FONT", fn_regular="Graphics/Knewave-Regular.ttf")


class MainApp(MDApp):

    def build(self):
        self.title = "CarFinder"

        Window.size = (600, 800)
        Window.top = 50
        Window.left = 50

        sm = MDScreenManager()

        sm.add_widget(MainScreen("mainscreen"))
        sm.add_widget(FinalScreen("finalscreen"))

        sm.current = "mainscreen"
        return sm


MainApp().run()
