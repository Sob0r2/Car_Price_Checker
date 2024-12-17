import os
import sys
from threading import Thread

from kivy.clock import Clock
from kivy.core.window import Window
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.screen import MDScreen

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "App")))

from Predict_Car_Price import final_evaluation


class MainScreen(MDScreen):
    """
    Main screen of the application for processing car price predictions.

    Attributes:
        dialog (MDDialog): A dialog box for displaying errors (e.g., invalid URLs).
    """

    dialog = None

    def show_dialog(self):
        """
        Display an error dialog when the provided URL is invalid.
        """
        if not self.dialog:
            self.dialog = MDDialog(
                title="Your URL is invalid",
                text="Please enter a valid URL.",
                md_bg_color=(0.8, 0.8, 0.8, 1),
                buttons=[
                    MDFlatButton(
                        text="OK",
                        theme_text_color="Custom",
                        text_color=(0, 0, 0, 1),
                        on_release=self.close_dialog,
                    ),
                ],
            )
        self.dialog.open()

    def on_button_click(self):
        """
        Handle the click event of the 'Search' button.
        Validates the URL and initiates processing in a separate thread.
        """
        url = self.ids.url_input.text.strip()
        if not url:
            self.show_dialog()
            return

        # Disable input and show loading spinner
        self.ids.url_input.opacity = 0
        self.ids.url_input.disabled = True
        self.ids.spinner.active = True

        # Start processing URL in a background thread
        thread = Thread(target=self.process_url, args=(url,))
        thread.start()

    def process_url(self, url):
        """
        Process the provided URL and update the application state.

        Args:
            url (str): The URL to a valid website containing the car whose price is to be predicted.

        Raises:
            Exception: If an error occurs during processing, it is caught,
            and the user is prompted to provide a new valid URL.
        """
        try:
            # Perform evaluation
            pred_price, current_price = final_evaluation(url)
            app = MDApp.get_running_app()
            app.pred_price = pred_price
            app.current_price = current_price

            # Notify completion and transition to final screen
            Clock.schedule_once(lambda dt: self.on_processing_complete(success=True))
            Clock.schedule_once(
                lambda dt: setattr(self.parent, "current", "finalscreen")
            )
        except Exception as e:
            print(f"Error: {e}")
            Clock.schedule_once(lambda dt: self.on_processing_complete(success=False))

    def on_processing_complete(self, success):
        """
        Callback invoked when URL processing is complete.

        Args:
            success (bool): Indicates whether the processing succeeded.
        """
        self.ids.spinner.active = False
        self.ids.url_input.opacity = 1
        self.ids.url_input.disabled = False
        self.ids.url_input.text = ""

        if not success:
            self.show_dialog()

    def close_dialog(self, *args):
        """
        Close the currently active error dialog.
        """
        if self.dialog:
            self.dialog.dismiss()

    def on_pre_leave(self):
        """
        Unbind the back key listener when leaving the screen.
        """
        Window.unbind(on_keyboard=self.on_key_down)

    def on_key_down(self, window, key, *largs):
        """
        Handle the back key press event.

        Args:
            window (Window): The Kivy window instance.
            key (int): The key code of the pressed key.
            largs: Additional arguments.
        """
        if key == 27:  # Back key
            app = MDApp.get_running_app()
            app.stop()
