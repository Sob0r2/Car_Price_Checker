
# Car Price Checker

This project allows you to predict car prices based on the car images and
data scrapped from otomoto.pl. Model is able to predict if some car is 
worth to buy or not

## How to Get Started

Follow these steps to get the project up and running on your local machine.

### 1. Clone the Repository

Clone the repository to your local machine using the following command:

```bash
git clone https://github.com/Sob0r2/Car_Price_Checker.git
cd Car_Price_Checker
```

### 2. Create a Virtual Environment

To create a virtual environment, use the `environment.yaml` file provided in the repository. This will set up the necessary dependencies for the project.

```bash
conda env create -f environment.yaml
conda activate car_project
```

### 3. Set Up the `.env` File

The project uses environment variables stored in a `.env` file. To set it up:

1. Copy the `.env.example` file to `.env`:

   ```bash
   cp .env.example .env
   ```

2. Open the `.env` file in a text editor and fill in the required paths according to the instructions in the file.

   The `.env` file include paths for datasets and model weights.

### 4. Run the Project

Once the environment is set up and the `.env` file is configured, you can start using the project. Follow the instructions in the project for specific usage or run the script directly to begin processing data.

```bash
cd App
python main.py
```

## Additional Information

For more detailed information, please refer to the documentation inside the code or check the individual script files.