import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision import transforms


class Car_Dataset(Dataset):
    """
    Custom PyTorch Dataset for handling car data that combines tabular features,
    image tensors, and labels.

    Attributes:
        data (pd.DataFrame): DataFrame containing the tabular and image data.
        image_col (str): Name of the column containing image tensors.
        label_col (str): Name of the column containing labels.
        device (str): Device to which tensors are sent (e.g., "cpu" or "cuda").
        transform (transforms.Compose): Transformations applied to the image tensors.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        image_col: str,
        label_col: str,
        device="cpu",
        transform=None,
    ):
        self.data = data_df
        self.image_col = image_col
        self.label_col = label_col
        self.device = device
        self.transform = transform

        self.tabular_columns = [
            col for col in self.data.columns if col not in [image_col, label_col]
        ]

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing tabular data, image tensor, and label.
        """
        row = self.data.iloc[idx]
        tabular_data = torch.tensor(row[self.tabular_columns], dtype=torch.float32).to(
            self.device
        )
        img = row[self.image_col]

        if self.transform:
            image = self.transform(img)

        image = image.to(self.device)
        label = torch.tensor(row[self.label_col], dtype=torch.float32).to(self.device)

        return {"tabular_data": tabular_data, "image": image, "label": label}


def process_image(file_path: str, folder: str) -> pd.DataFrame:
    """
    Processes an image file into a DataFrame with a tensor.

    Args:
        file_path (str): Path to the image file.
        folder (str): Folder name for grouping.

    Returns:
        pd.DataFrame: DataFrame containing the image tensor and folder info.
    """
    try:
        image = Image.open(file_path).convert("RGB")
        return pd.DataFrame([{"folder": folder, "image_tensor": image}])
    except Exception as e:
        print(f"Cannot process image: {file_path}: {e}")
        return pd.DataFrame()


def process_tabular(file_path: str, folder: str) -> pd.DataFrame:
    """
    Processes a JSON file into a tabular DataFrame.

    Args:
        file_path (str): Path to the JSON file.
        folder (str): Folder name for grouping.

    Returns:
        pd.DataFrame: DataFrame containing the tabular data and folder info.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame([data])
        df["folder"] = folder
        return df
    except Exception as e:
        print(f"Cannot process tabular data: {file_path}: {e}")
        return pd.DataFrame()


def get_only_image_data(path: str) -> pd.DataFrame:
    """
    Retrieves image data from the given directory.

    Args:
        path (str): Directory path containing image files.

    Returns:
        pd.DataFrame: DataFrame containing image tensors and folder information.
    """
    results = pd.DataFrame([])

    with ThreadPoolExecutor() as executor:
        futures = []
        for folder in os.listdir(path):
            if not folder.endswith(("txt")):
                folder_path = os.path.join(path, folder)
                for file in os.listdir(folder_path):
                    if file.endswith(("jpg", "jpeg", "png", "bmp")):
                        file_path = os.path.join(folder_path, file)
                        futures.append(
                            executor.submit(process_image, file_path, folder)
                        )

        for future in futures:
            results = pd.concat([results, future.result()], ignore_index=True)

    return results


def get_only_tabular_data(path: str) -> pd.DataFrame:
    """
    Retrieves tabular data from the given directory.

    Args:
        path (str): Directory path containing JSON files.

    Returns:
        pd.DataFrame: DataFrame containing tabular data and folder information.
    """
    results = pd.DataFrame([])

    with ThreadPoolExecutor() as executor:
        futures = []
        for folder in os.listdir(path):
            if not folder.endswith(("txt")):
                folder_path = os.path.join(path, folder)
                for file in os.listdir(folder_path):
                    if file.endswith(("json")):
                        file_path = os.path.join(folder_path, file)
                        futures.append(
                            executor.submit(process_tabular, file_path, folder)
                        )

        for future in futures:
            results = pd.concat([results, future.result()], ignore_index=True)

    return results


def preprocess_tabular_data(tabular_df):
    tabular_df = tabular_df.drop("nr_seats", axis=1)
    if "link" in tabular_df.columns:
        tabular_df = tabular_df.drop("nr_seats", axis=1)
    tabular_df.loc[tabular_df["fuel_type"] == "Elektryczny", "engine displacement"] = 0

    door_mapper = {"2": "3", "4": "5"}
    tabular_df["door_count"] = tabular_df["door_count"].apply(
        lambda x: door_mapper.get(x, x)
    )

    tabular_df = tabular_df.replace("", np.nan)
    tabular_df = tabular_df.dropna()

    tabular_df["price"] = tabular_df["price"].apply(
        lambda x: float(x.replace(" ", "").replace(",", "."))
    )
    tabular_df["door_count"] = tabular_df["door_count"].apply(lambda x: int(x))
    tabular_df["year"] = pd.to_numeric(tabular_df["year"], errors="coerce")
    tabular_df["mileage"] = tabular_df["mileage"].apply(
        lambda x: int(x.split(" km")[0].replace(" ", ""))
    )
    tabular_df["engine displacement"] = tabular_df["engine displacement"].apply(
        lambda x: int(x.split(" cm3")[0].replace(" ", "")) if x != 0 else 0
    )
    tabular_df["engine power"] = tabular_df["engine power"].apply(
        lambda x: int(x.split(" KM")[0].replace(" ", ""))
    )

    tabular_df = pd.get_dummies(
        tabular_df,
        columns=["fuel_type", "transmission", "car_model", "body type", "new_or_use"],
        drop_first=True,
    )

    return tabular_df


def create_dataset(
    train: bool = True,
    transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
    eda=False,
    scaler=None,
) -> Car_Dataset:
    """
    Creates a PyTorch Dataset for car data.

    Args:
        train (bool): Whether to load training or testing data.
        transform (transforms.Compose): Transformations for image data.
        eda (bool): Whether to return data for exploratory data analysis.

    Returns:
        CarDataset: Dataset containing combined image and tabular data.
    """
    path = (
        os.getenv("FINAL_DATASET") + "train"
        if train
        else os.getenv("FINAL_DATASET") + "test"
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    image_df = get_only_image_data(path)
    tabular_df = get_only_tabular_data(path)

    if eda:
        return image_df.merge(tabular_df, how="left", on="folder")

    tabular_df = preprocess_tabular_data(tabular_df)

    if train:
        scaler = MinMaxScaler()
        columns_to_scale = tabular_df.columns.difference(["folder", "price"])
        tabular_df = tabular_df.sort_index(axis=1)
        tabular_df[columns_to_scale] = scaler.fit_transform(
            tabular_df[columns_to_scale]
        )
        dump(scaler, (os.path.join(os.getenv("WEIGHTS"), "minmax_scaler.joblib")))
        dump(tabular_df.columns, (os.path.join(os.getenv("WEIGHTS"), "columns.joblib")))
    else:
        columns = load(os.path.join(os.getenv("WEIGHTS"), "columns.joblib"))
        scaler = load(os.path.join(os.getenv("WEIGHTS"), "minmax_scaler.joblib"))

        for col in columns:
            if col not in tabular_df.columns:
                tabular_df[col] = 0

        columns_to_scale = tabular_df.columns.difference(["folder", "price"])
        tabular_df = tabular_df.sort_index(axis=1)
        tabular_df[columns_to_scale] = scaler.transform(tabular_df[columns_to_scale])

    final_df = (
        image_df.merge(tabular_df, how="left", on="folder")
        .drop(columns=["folder"])
        .dropna()
    )

    return Car_Dataset(final_df, "image_tensor", "price", device, transform)
