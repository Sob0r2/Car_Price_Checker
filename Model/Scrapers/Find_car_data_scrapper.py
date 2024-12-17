import asyncio
import os
from io import BytesIO

import aiohttp
import requests
from bs4 import BeautifulSoup
from PIL import Image

MAIN_PAGE_URL = "https://www.otomoto.pl/osobowe?search%5Bfilter_enum_damaged%5D=0"


class Car_Photos_Dataset:
    """
    A class to scrape car images from the 'Otomoto' website and create a dataset.
    The data will be used to train model to predict if there is a car in the image

    Attributes:
        car_pages_links (list): A list to store URLs of car pages.
        data_folder (str): Path to the directory where the images will be saved.
        headers (dict): HTTP headers to be used in requests.
    """

    def __init__(self) -> None:
        self.car_pages_links = []
        self.data_folder = os.getenv("CAR_OR_NOT")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

    def get_car_links(self, n: int = 200) -> None:
        """
        Scrapes car page links from the 'Otomoto' website, stopping after 'n' links are collected.

        Args:
            n (int): The number of car pages to scrape.
        """
        start_page = 1
        car_ctr = 0

        while car_ctr < n:
            link = MAIN_PAGE_URL + f"&page={start_page}"
            start_page += 1

            response = requests.get(link, headers=self.headers)
            soup = BeautifulSoup(response.text, "html.parser")

            all_links = soup.find_all("h1", class_="epwfahw9 ooa-1ed90th er34gjf0")
            for item in all_links:
                if car_ctr >= n:
                    break

                link_tag = item.find("a")
                if link_tag and link_tag.get("href"):
                    self.car_pages_links.append(link_tag["href"])
                    car_ctr += 1
                    print(f"{car_ctr}: {link_tag['href']}")

    def create_train_and_test_set(self) -> None:
        """
        Creates training and testing datasets by downloading car images from scraped links.

        The images are split into training and testing folders based on the order of the links.
        """
        train_data = os.path.join(self.data_folder, "Car_photos_train")
        test_data = os.path.join(self.data_folder, "Car_photos_test")
        n = len(self.car_pages_links)

        asyncio.run(self.async_download_images(n, train_data, test_data))

    async def async_download_images(
        self, n: int, train_data: str, test_data: str
    ) -> None:
        """
        Asynchronously downloads car images from the scraped links and saves \
        them to the appropriate directories.

        Args:
            n (int): The number of car pages to process.
            train_data (str): The path to the directory for saving training images.
            test_data (str): The path to the directory for saving testing images.
        """
        sem = asyncio.Semaphore(8)
        async with aiohttp.ClientSession(
            headers=self.headers, timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            tasks = []
            img_number = 0

            for idx, link in enumerate(self.car_pages_links):
                response = requests.get(link, headers=self.headers)
                soup = BeautifulSoup(response.text, "html.parser")
                photos = soup.find_all("div", class_="ooa-12np8kw e142atj30")

                for photo_link in photos:
                    photo = photo_link.find("img")
                    photo_url = photo.get("src")
                    if photo_url:
                        save_folder = train_data if idx <= n // 2 else test_data
                        tasks.append(
                            self.download_image(
                                session, photo_url, img_number, save_folder, sem
                            )
                        )
                        img_number += 1

            await asyncio.gather(*tasks)

    async def download_image(
        self,
        session: aiohttp.ClientSession,
        url: str,
        image_number: int,
        save_folder: str,
        sem: asyncio.Semaphore,
        target_size: tuple = (224, 224),
        max_retries: int = 3,
    ) -> None:
        """
        Downloads a single image from the given URL and saves it to the specified folder.

        Args:
            session (aiohttp.ClientSession): The aiohttp session used for making requests.
            url (str): The URL of the image to download.
            image_number (int): The number of the image, used for naming the file.
            save_folder (str): The folder where the image will be saved.
            sem (asyncio.Semaphore): Semaphore for controlling concurrency.
            target_size (tuple): The target size for resizing the image.
            max_retries (int): The maximum number of retry attempts in case of failure.
        """
        async with sem:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            for attempt in range(max_retries):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.read()
                            img = Image.open(BytesIO(content))

                            img_resized = img.resize(target_size)
                            file_name = os.path.join(
                                save_folder, f"image_{image_number}.jpg"
                            )

                            img_resized.save(file_name, format="JPEG")
                            print(f"Image saved and resized as {file_name}")
                            break
                        else:
                            print(
                                f"Failed to download image {image_number}. \
                                Status code: {response.status}"
                            )
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    print(f"Error downloading image {image_number}: {e}")
                    if attempt < max_retries - 1:
                        print(
                            f"Retrying {image_number}... ({attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(2)
                    else:
                        print(
                            f"Failed to download image {image_number} after {max_retries} attempts"
                        )
