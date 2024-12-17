import json
import os

from kivy.uix.image import Image
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen

#  Set to store dynamically added widgets
dynamic_widgets = set()


class FinalScreen(MDScreen):
    """
    Final screen for displaying car details and price evaluation.

    This screen shows the car's image, details (e.g., model, year, mileage),
    and a comparison of its predicted price with the actual price. It also
    evaluates whether the car is worth buying based on the price comparison.
    """

    def on_enter(self, *args):
        """
        Initialize the final screen by loading car data and dynamically
        updating the user interface.

        Car details are fetched from a JSON file, while the car image is
        fetched from the first picture of the car used to predict price

        Args:
            *args: Variable length argument list (passed automatically by Kivy).
        """
        app = MDApp.get_running_app()
        self.predicted_price = app.pred_price
        self.real_price = app.current_price

        # Load data from JSON files in the current directory
        path = os.getenv("CURRENT")
        for file in os.listdir(path):
            if file.endswith(".json"):
                with open(os.path.join(path, file), "r") as f:
                    content = json.load(f)
                # Extract car details
                self.car_model = content.get("car_model", "Unknown")
                self.year = str(content.get("year", "Unknown"))
                self.mileage = str(content.get("mileage", "Unknown"))
                self.engine_power = str(content.get("engine power", "Unknown"))
                self.transmission = content.get("transmission", "Unknown")
                self.fuel_type = content.get("fuel_type", "Unknown")
                break

        # Add the car image to the screen
        self.car_image = Image(
            source=os.path.join(path, "0.jpg"),
            size=(400, 400),
            pos_hint={"center_x": 0.24, "center_y": 0.52},
        )
        self.car_image.reload()  # Reload the image in case of updates
        self.add_widget(self.car_image)
        dynamic_widgets.add(self.car_image)

        # Determine whether the car is worth buying based on price comparison
        self.worth_to_buy = (
            "WORTH TO BUY"
            if self.predicted_price >= self.real_price
            else "NOT WORTH TO BUY"
        )
        self.color = (
            (0, 1, 0, 1) if self.worth_to_buy == "WORTH TO BUY" else (1, 0, 0, 1)
        )

        # Add worth-to-buy evaluation label
        worth_to_buy = MDLabel(
            text=self.worth_to_buy,
            theme_text_color="Custom",
            text_color=self.color,
            outline_color=(0, 0, 0, 1),
            outline_width=2,
            halign="center",
            pos_hint={"center_x": 0.5, "center_y": 0.75},
        )
        worth_to_buy.font_size = 50
        worth_to_buy.font_name = "TITLE_FONT"
        self.add_widget(worth_to_buy)
        dynamic_widgets.add(worth_to_buy)

        # Add car details as labels
        info_labels = [
            f"Car Model: {self.car_model}",
            f"Year: {self.year}",
            f"Mileage: {self.mileage}",
            f"Engine Power: {self.engine_power}",
            f"Transmission: {self.transmission}",
            f"Fuel Type: {self.fuel_type}",
        ]
        y_position = 0.65
        for info in info_labels:
            label = MDLabel(
                text=info,
                theme_text_color="Custom",
                text_color=(0.149, 0.898, 0.99, 1),
                outline_color=(0, 0, 0, 1),
                outline_width=2,
                pos_hint={"center_x": 0.98, "center_y": y_position},
            )
            label.font_size = 20
            label.font_name = "TEXT_FONT"
            self.add_widget(label)
            dynamic_widgets.add(label)
            y_position -= 0.05

        # Add real price label
        real_price_widget = MDLabel(
            text=f"Price on site: {self.real_price} PLN",
            theme_text_color="Custom",
            text_color=(0.796, 0.029, 0.441, 0.8),
            outline_color=(0, 0, 0, 1),
            outline_width=2,
            halign="left",
            pos_hint={"center_x": 0.55, "center_y": 0.33},
        )
        real_price_widget.font_size = 25
        real_price_widget.font_name = "TEXT_FONT"
        self.add_widget(real_price_widget)
        dynamic_widgets.add(real_price_widget)

        # Add predicted price label
        predicted_price_widget = MDLabel(
            text=f"Predicted price: {self.predicted_price} PLN",
            theme_text_color="Custom",
            text_color=(0.796, 0.029, 0.441, 0.8),
            outline_color=(0, 0, 0, 1),
            outline_width=2,
            halign="left",
            pos_hint={"center_x": 0.55, "center_y": 0.27},
        )
        predicted_price_widget.font_size = 25
        predicted_price_widget.font_name = "TEXT_FONT"
        self.add_widget(predicted_price_widget)
        dynamic_widgets.add(predicted_price_widget)

    def clear_dynamic_widgets(self):
        """
        Clear all dynamically added widgets from the screen.

        This ensures the screen is refreshed properly when revisited.
        """
        for widget in dynamic_widgets:
            self.remove_widget(widget)
        dynamic_widgets.clear()

    def on_button_click(self, *args):
        """
        Handle the 'Back' button click.

        Clears dynamic widgets and navigates back to the main screen.

        Args:
            *args: Variable length argument list (passed automatically by Kivy).
        """
        self.clear_dynamic_widgets()
        self.manager.current = "mainscreen"
