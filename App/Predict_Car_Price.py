import asyncio
import json
import os
import sys
from io import BytesIO

import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from joblib import load
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.Architectures.CarPricePredictionModel import CarPricePredictionModel
from Model.Car_Dataset import Car_Dataset, preprocess_tabular_data
from utils import get_model, scrap_car_tabular_data

# Constants
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
    (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def scrap_single_car_data(url):
    """
    Scrapes data for a single car from a given URL.

    :param url: URL of the car listing.
    :return: A tuple containing a DataFrame with car details and a list of car images as tensors.
    """
    car_or_not_model = get_model()

    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    images_ls = []

    # Extract car information and save to a JSON file
    car_info = asyncio.run(scrap_car_tabular_data(soup))
    json_file_path = os.path.join(os.getenv("CURRENT"), f"{car_info['car_model']}.json")
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(car_info, json_file, ensure_ascii=False, indent=4)

    # Extract car images
    photos = soup.find_all("div", class_="ooa-12np8kw e142atj30")
    image_number = 0
    for photo_link in photos:
        photo = photo_link.find("img")
        photo_url = photo.get("src")

        # Process image
        content = requests.get(photo_url).content
        img = Image.open(BytesIO(content))
        img_resized = img.resize((224, 224))

        # Convert to tensor
        to_tensor = transforms.ToTensor()
        tensor_image = to_tensor(img_resized).to(DEVICE)

        # Predict if the image is relevant (car or not)
        output = car_or_not_model(tensor_image.unsqueeze(0))
        softmax_output = torch.softmax(output, dim=1)
        label = 1 if softmax_output[0, 1].item() >= 0.5 else 0

        # Save relevant images
        if label:
            images_ls.append(tensor_image.cpu())
            file_name = os.path.join(os.getenv("CURRENT"), f"{image_number}.jpg")
            img_resized.save(file_name, format="JPEG")
            print(f"Image saved and resized as {file_name}")
            image_number += 1

    return pd.DataFrame([car_info]), images_ls


def final_evaluation(url):
    """
    Performs the final evaluation of the car based on its tabular and image data.

    :param url: URL of the car listing.
    :return: A tuple containing the predicted price and actual price of the car.
    """
    # Clear current working directory
    for file_name in os.listdir(os.getenv("CURRENT")):
        file_path = os.path.join(os.getenv("CURRENT"), file_name)
        os.remove(file_path)

    # Scrap car data
    tabular_df, images_ls = scrap_single_car_data(url)

    # Load the price prediction model
    model = CarPricePredictionModel(29)
    model.load_state_dict(
        torch.load(os.path.join(os.getenv("WEIGHTS"), "Final_Model.pth"))
    )
    model.eval()
    model = model.to(DEVICE)

    # Preprocess tabular data
    tabular_df = preprocess_tabular_data(tabular_df)
    columns = load(os.path.join(os.getenv("WEIGHTS"), "columns.joblib"))
    scaler = load(os.path.join(os.getenv("WEIGHTS"), "minmax_scaler.joblib"))

    # Ensure all required columns are present
    for col in columns:
        if col not in tabular_df.columns:
            tabular_df[col] = 0

    columns_to_scale = tabular_df.columns.difference(["price", "folder"])
    tabular_df = tabular_df.sort_index(axis=1)
    tabular_df[columns_to_scale] = scaler.transform(tabular_df[columns_to_scale])

    # Create final DataFrame for all images
    final_df = pd.concat([tabular_df] * len(images_ls), ignore_index=True)
    final_df["image"] = [ToPILImage()(image) for image in images_ls]
    final_df = final_df.dropna().drop("folder", axis=1)

    test_transformations = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data = Car_Dataset(final_df, "image", "price", DEVICE, test_transformations)
    loader = DataLoader(data, batch_size=len(data), num_workers=0)

    with torch.no_grad():
        for dict in loader:
            tabular_data, images, label = (
                dict["tabular_data"].to(DEVICE),
                dict["image"].to(DEVICE),
                dict["label"].to(DEVICE),
            )
            output = model(images, tabular_data)

    # Return predicted and actual prices
    return round(torch.mean(output).item(), 2), round(torch.mean(label).item(), 2)
