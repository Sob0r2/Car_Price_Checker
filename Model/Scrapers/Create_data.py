import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import json
import time
from io import BytesIO

import aiohttp
import requests
import torch
from bs4 import BeautifulSoup
from PIL import Image
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from torchvision import transforms
from webdriver_manager.chrome import ChromeDriverManager

from utils import get_model, scrap_car_tabular_data

CAR_MODELS = [
    "BMW",
    "Audi",
    "Volkswagen",
    "Ford",
    "Opel",
    "Toyota",
    "Skoda",
    "Renault",
    "Peugeot",
]
MAIN_PAGE_URL = "https://www.otomoto.pl/osobowe/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class CarLinksScraper:
    """
    A class to scrape car links from the otomoto.pl website.
    It collects car data and save them to the proper place
    """

    def __init__(self, train=True) -> None:
        self.car_pages_links = []
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        self.car_urls = [
            MAIN_PAGE_URL + car_name + "?search%5Bfilter_enum_damaged%5D=0"
            for car_name in CAR_MODELS
        ]
        self.car_data_path = (
            os.getenv("FINAL_DATASET") + "train"
            if train
            else os.getenv("FINAL_DATASET") + "test"
        )
        self.train = train
        self.driver = self.init_selenium_driver()

    def init_selenium_driver(self):
        """
        Initializes the Selenium WebDriver for scraping the pages.

        Returns:
            webdriver.Chrome: The initialized WebDriver.
        """
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"user-agent={self.headers['User-Agent']}")
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )
        return driver

    def get_car_links(self, n, start):
        """
        Retrieves car links from various result pages for the selected car models.

        Args:
            n (int): Number of pages to scrape.
            start (int): The starting page number.
        """
        car_pages_links = []
        for idx, car_model_url in enumerate(self.car_urls):

            for page in range(n):
                link = f"{car_model_url}&page={start + page + 1}"
                response = requests.get(link, headers=self.headers)
                soup = BeautifulSoup(response.text, "html.parser")

                all_links = soup.find_all("h1", class_="epwfahw9 ooa-1ed90th er34gjf0")
                car_pages_links += [
                    item.find("a")["href"]
                    for item in all_links
                    if item.find("a") and item.find("a").get("href")
                ]

        filepath = os.path.join(self.car_data_path, "car_links.txt")
        with open(filepath, "w") as file:
            file.writelines(f"{link}\n" for link in car_pages_links)

    def get_car_page_with_selenium(self, url, retries=3):
        """
        Fetches the car page using Selenium.

        Args:
            url (str): The URL of the car's page.
            retries (int): Number of retries in case of failure.

        Returns:
            str: The source code of the page.
        """
        for attempt in range(retries):
            try:
                self.driver.get(url)
                self.driver.implicitly_wait(3)
                return self.driver.page_source
            except selenium.common.exceptions.WebDriverException as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < retries - 1:
                    time.sleep(2)  # Wait before retrying
                else:
                    raise  # If retries are exhausted, raise the exception

    def create_dataset(self, folder_i=0):
        """
        Creates the dataset (images and tabular data) for the car links.

        Args:
            folder_i (int): The folder index for saving data.
        """
        self.model = get_model()
        filepath = os.path.join(self.car_data_path, "car_links.txt")

        with open(filepath, "r") as file:
            car_pages_links = file.readlines()

        car_pages_links = [link.strip() for link in car_pages_links if link.strip()]
        asyncio.run(self.download_all_data(car_pages_links, folder_i))

    async def download_all_data(self, links, folder_i):
        """
        Asynchronously downloads car data (images and tabular data) for the provided links.

        Args:
            links (list): List of car page URLs.
            folder_i (int): Folder index for saving data.
        """
        sem = asyncio.Semaphore(4)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            tasks = []
            for idx, link in enumerate(links):
                car_model = next(
                    (car for car in CAR_MODELS if car.lower() in link.lower()), ""
                )
                if not car_model:
                    continue

                print(f"Processing: {link}")
                if "otomoto" in link:
                    page_source = self.get_car_page_with_selenium(link)
                    soup = BeautifulSoup(page_source, "html.parser")
                else:
                    async with sem:
                        response = await session.get(link, headers=self.headers)
                        soup = BeautifulSoup(await response.text(), "html.parser")

                file_name = (
                    f"{car_model}_{1000 * folder_i + idx}" if self.train else "current"
                )
                folder_path = os.path.join(self.car_data_path, file_name)
                os.makedirs(folder_path, exist_ok=True)

                tasks.append(
                    asyncio.create_task(
                        self.process_car_data(
                            session, soup, sem, folder_path, car_model, link
                        )
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    print(f"Error during processing: {result}")

    async def process_car_data(self, session, soup, sem, folder_path, car_model, link):
        """
        Processes the car data, including downloading images and saving tabular data.

        Args:
            session (aiohttp.ClientSession): The HTTP session.
            soup (BeautifulSoup): BeautifulSoup object containing the car page.
            sem (asyncio.Semaphore): Semaphore for controlling concurrent tasks.
            folder_path (str): Folder path where data will be saved.
            car_model (str): The car model.
            link (str): The URL of the car's page.
        """
        await self.download_images(session, soup, sem, folder_path)
        await self.download_tabular_data(soup, sem, folder_path, car_model, link)

    async def download_images(
        self, session, soup, sem, folder_path, target_size=(224, 224), max_retries=3
    ):
        """
        Downloads images from the page and saves them to the specified folder.

        Args:
            session (aiohttp.ClientSession): The HTTP session.
            soup (BeautifulSoup): BeautifulSoup object containing the car page.
            sem (asyncio.Semaphore): Semaphore to control the number of concurrent tasks.
            folder_path (str): The folder where images will be saved.
            target_size (tuple): The target image size.
            max_retries (int): Maximum number of retries for downloading images.
        """
        async with sem:
            image_number = 0
            photos = soup.find_all("div", class_="ooa-12np8kw e142atj30")

            for photo_link in photos:
                photo = photo_link.find("img")
                photo_url = photo.get("src")

                for attempt in range(max_retries):
                    try:
                        async with session.get(photo_url) as response:
                            if response.status == 200:
                                content = await response.read()

                                img = Image.open(BytesIO(content))
                                img_resized = img.resize(target_size)

                                to_tensor = transforms.ToTensor()
                                tensor_image = to_tensor(img_resized)
                                tensor_image = tensor_image.to(DEVICE)

                                output = self.model(tensor_image.unsqueeze(0))
                                softmax_output = torch.softmax(output, dim=1)
                                label = 1 if softmax_output[0, 1].item() >= 0.7 else 0

                                if label:
                                    file_name = os.path.join(
                                        folder_path, f"image_{image_number}.jpg"
                                    )

                                    img_resized.save(file_name, format="JPEG")
                                    print(f"Image saved and resized as {file_name}")
                                    image_number += 1
                                break
                            else:
                                print(
                                    f"Failed to download image. Status code: {response.status}"
                                )
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        print(f"Error downloading image: {e}")
                        if attempt < max_retries - 1:
                            print(f"Retrying... ({attempt + 1}/{max_retries})")
                            await asyncio.sleep(2)
                        else:
                            print("Failed to download image after retries.")

    async def download_tabular_data(self, soup, sem, folder_path, car_model, link):
        """
        Collects tabular data about the car and saves it in a JSON file.

        Args:
            soup (BeautifulSoup): BeautifulSoup object containing the car page.
            sem (asyncio.Semaphore): Semaphore for controlling concurrent tasks.
            folder_path (str): Folder where the data will be saved.
            car_model (str): The car model.
            link (str): The URL of the car's page.
        """
        async with sem:
            # Scrap the car's tabular data (e.g., specifications, price, etc.)
            car_info = await scrap_car_tabular_data(soup)
            car_info["car_model"] = car_model
            car_info["link"] = link

            # Save the tabular data to a JSON file
            json_file_path = os.path.join(folder_path, f"{car_model}.json")
            with open(json_file_path, "w", encoding="utf-8") as json_file:
                json.dump(car_info, json_file, ensure_ascii=False, indent=4)

            print(f"Saved data to the file: {json_file_path}")

    def close(self):
        self.driver.quit()
