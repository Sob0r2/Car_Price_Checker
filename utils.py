import os

import torch
from torchvision import models


async def scrap_car_tabular_data(soup):
    """
    Extracts car details such as car model, year, mileage, fuel type, etc.,
    from the provided BeautifulSoup object of a car listing page.

    Args:
        soup (BeautifulSoup): Parsed HTML content of a car listing page.

    Returns:
        dict: A dictionary containing extracted car details.
    """
    basic_elements = ["make", "year", "door_count", "nr_seats"]
    car_info = {}

    for stat in basic_elements:
        div = soup.find("div", {"data-testid": stat})
        if div:
            ele = div.find("p", class_="eim4snj8 ooa-17xeqrd")
            if ele:
                # Special handling for "make" to map it to "car_model"
                if stat == "make":
                    car_info["car_model"] = ele.get_text(strip=True)
                else:
                    car_info[stat] = ele.get_text(strip=True)
            else:
                car_info[stat] = ""
        else:
            car_info[stat] = ""

    detailed_elements = {
        "Przebieg": "mileage",
        "Rodzaj paliwa": "fuel_type",
        "Skrzynia biegów": "transmission",
        "Typ nadwozia": "body type",
        "Pojemność skokowa": "engine displacement",
        "Moc": "engine power",
    }

    for stat, key in detailed_elements.items():
        div = soup.find("div", {"aria-label": lambda x: x and stat in x})
        if div:
            car_info[key] = div.find("p", class_="ee3fiwr2 ooa-1rcllto").text
        else:
            car_info[key] = ""

    div = soup.find("div", class_="ooa-1821gv5 e12csvfg1")
    if div:
        car_info["new_or_use"] = div.find("p", class_="e7ig7db0 ooa-vy37q4").text.split(
            " "
        )[0]
    else:
        car_info["new_or_use"] = ""

    div = soup.find("div", class_="ooa-18a76w7 evnmei42")
    if div:
        car_info["price"] = div.find("h3").text
    else:
        car_info["price"] = ""

    return car_info


def get_model():
    """
    Loads a pre-trained model to classify images as either "car" or "not car."
    An image is classified as a "car" if it shows an external view of a vehicle
    where the full geometry of the car is visible.

    Returns:
        torch.nn.Module: The loaded model configured for the current device (CPU or GPU).
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = models.vgg16_bn(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 2)
    model.load_state_dict(
        torch.load(os.path.join(os.getenv("WEIGHTS"), "car_or_not_model.pth"))
    )
    model = model.to(device)
    return model
