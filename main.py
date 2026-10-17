from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
import os
import urllib.request
import pandas as pd
import joblib
import json
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import URLError, HTTPError


# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Smart Inventory Replenishment API",
    description="AI-Based Inventory Demand Prediction and Replenishment System",
    version="2.0"
)


# ==========================================
# HTML TEMPLATE CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(
    directory=str(BASE_DIR / "template")
)


# ==========================================
# LOAD ML MODEL
# ==========================================

MODEL_PATH = BASE_DIR / "improved_demand_prediction_model.pkl"
MODEL_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1I63DgXwzVPuAU4O_0jnTREhtlJ2PmcUf](https://drive.google.com/uc?export=download&id=1I63DgXwzVPuAU4O_0jnTREhtlJ2PmcUf)"

if not MODEL_PATH.exists():
    print("Model file not found locally. Downloading from cloud storage...")
    try:
        # Downloads the file automatically if it's missing (e.g., when deployed on a cloud server or fresh clone)
        urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, MODEL_PATH)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")

# Load the model
try:
    model = joblib.load(MODEL_PATH)
    print("Machine Learning Model Loaded Successfully!")
except Exception as e:
    model = None
    print(f"Warning: Could not load model. Error: {e}")


# ==========================================
# HOME PAGE
# ==========================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "loaded",
        "weather_api": "Open-Meteo"
    }


# ==========================================
# OPEN-METEO WEATHER CODE CONVERSION
# ==========================================

def weather_code_to_condition(code):

    """
    Convert Open-Meteo WMO weather codes
    into the categories already used by
    the machine-learning model.
    """

    code = int(code)

    # Clear / mainly clear
    if code in [0, 1]:
        return "Sunny"

    # Partly cloudy / overcast
    elif code in [2, 3]:
        return "Cloudy"

    # Fog
    elif code in [45, 48]:
        return "Cloudy"

    # Drizzle / rain / showers
    elif code in [
        51, 53, 55,
        56, 57,
        61, 63, 65,
        66, 67,
        80, 81, 82
    ]:
        return "Rainy"

    # Snow
    elif code in [
        71, 73, 75, 77,
        85, 86
    ]:
        return "Snowy"

    # Thunderstorm
    elif code in [95, 96, 99]:
        return "Rainy"

    return "Cloudy"


# ==========================================
# WEATHER DESCRIPTION
# ==========================================

def weather_description(code):

    descriptions = {

        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",

        45: "Fog",
        48: "Depositing rime fog",

        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",

        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",

        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",

        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",

        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Thunderstorm with heavy hail"
    }

    return descriptions.get(
        int(code),
        "Unknown weather condition"
    )


# ==========================================
# OPEN-METEO GEOCODING
# ==========================================

def get_coordinates(city):

    params = urlencode({
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    })

    url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        + params
    )

    try:

        with urlopen(url, timeout=10) as response:

            data = json.loads(
                response.read().decode("utf-8")
            )

    except (URLError, HTTPError, TimeoutError) as error:

        raise HTTPException(
            status_code=503,
            detail=f"Unable to connect to weather location service: {error}"
        )

    results = data.get("results", [])

    if not results:

        raise HTTPException(
            status_code=404,
            detail=f"Location '{city}' was not found."
        )

    location = results[0]

    return {
        "name": location.get("name"),
        "country": location.get("country"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "timezone": location.get("timezone")
    }


# ==========================================
# OPEN-METEO CURRENT WEATHER
# ==========================================

def get_weather(latitude, longitude):

    params = urlencode({

        "latitude": latitude,
        "longitude": longitude,

        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "weather_code",
            "cloud_cover",
            "wind_speed_10m"
        ]),

        "timezone": "auto"

    })

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        + params
    )

    try:

        with urlopen(url, timeout=10) as response:

            data = json.loads(
                response.read().decode("utf-8")
            )

    except (URLError, HTTPError, TimeoutError) as error:

        raise HTTPException(
            status_code=503,
            detail=f"Unable to connect to Open-Meteo: {error}"
        )

    current = data.get("current", {})

    weather_code = current.get("weather_code")

    if weather_code is None:

        raise HTTPException(
            status_code=503,
            detail="Weather data was not returned by Open-Meteo."
        )

    condition = weather_code_to_condition(
        weather_code
    )

    return {

        "temperature": current.get(
            "temperature_2m"
        ),

        "humidity": current.get(
            "relative_humidity_2m"
        ),

        "apparent_temperature": current.get(
            "apparent_temperature"
        ),

        "precipitation": current.get(
            "precipitation"
        ),

        "rain": current.get(
            "rain"
        ),

        "weather_code": weather_code,

        "weather_description":
            weather_description(weather_code),

        "weather_condition":
            condition,

        "cloud_cover":
            current.get("cloud_cover"),

        "wind_speed":
            current.get("wind_speed_10m"),

        "time":
            current.get("time"),

        "timezone":
            data.get("timezone")

    }


# ==========================================
# WEATHER API ENDPOINT
# ==========================================

@app.get("/weather")
def weather(city: str):

    location = get_coordinates(city)

    weather_data = get_weather(
        location["latitude"],
        location["longitude"]
    )

    return {

        "location": location,

        "weather": weather_data,

        "source": "Open-Meteo",

        "status": "connected"

    }


# ==========================================
# PREDICTION INPUT MODEL
# ==========================================

class InventoryInput(BaseModel):

    store_id: str

    product_id: str

    category: str

    region: str

    inventory_level: float

    price: float

    discount: float

    weather_condition: str

    holiday_promotion: int

    competitor_pricing: float

    seasonality: str

    lag_1: float

    lag_7: float

    rolling_mean_7: float


# ==========================================
# PREDICTION API
# ==========================================

@app.post("/predict")
def predict_inventory(data: InventoryInput):

    # ======================================
    # DATE FEATURES
    # ======================================

    current_date = pd.Timestamp.now()

    year = current_date.year
    month = current_date.month
    day = current_date.day

    day_of_week = current_date.dayofweek

    is_weekend = (
        1 if day_of_week >= 5 else 0
    )


    # ======================================
    # PRICE FEATURES
    # ======================================

    price_difference = (
        data.price -
        data.competitor_pricing
    )

    if data.competitor_pricing != 0:

        price_ratio = (
            data.price /
            data.competitor_pricing
        )

    else:

        price_ratio = 1


    # ======================================
    # CREATE MODEL INPUT
    # ======================================

    input_data = pd.DataFrame([{

        "Store ID":
            data.store_id,

        "Product ID":
            data.product_id,

        "Category":
            data.category,

        "Region":
            data.region,

        "Inventory Level":
            data.inventory_level,

        "Price":
            data.price,

        "Discount":
            data.discount,

        "Weather Condition":
            data.weather_condition,

        "Holiday/Promotion":
            data.holiday_promotion,

        "Competitor Pricing":
            data.competitor_pricing,

        "Seasonality":
            data.seasonality,

        "Year":
            year,

        "Month":
            month,

        "Day":
            day,

        "DayOfWeek":
            day_of_week,

        "IsWeekend":
            is_weekend,

        "Price Difference":
            price_difference,

        "Price Ratio":
            price_ratio,

        "Lag_1":
            data.lag_1,

        "Lag_7":
            data.lag_7,

        "Rolling_Mean_7":
            data.rolling_mean_7

    }])


    # ======================================
    # ML DEMAND PREDICTION
    # ======================================

    predicted_demand = model.predict(
        input_data
    )[0]

    predicted_demand = max(
        0,
        float(predicted_demand)
    )


    # ======================================
    # INVENTORY PARAMETERS
    # ======================================

    LEAD_TIME = 2

    SAFETY_STOCK_PERCENTAGE = 0.20


    # ======================================
    # LEAD TIME DEMAND
    # ======================================

    lead_time_demand = (
        predicted_demand *
        LEAD_TIME
    )


    # ======================================
    # SAFETY STOCK
    # ======================================

    safety_stock = (
        predicted_demand *
        SAFETY_STOCK_PERCENTAGE
    )


    # ======================================
    # REORDER POINT
    # ======================================

    reorder_point = (
        lead_time_demand +
        safety_stock
    )


    # ======================================
    # REQUIRED INVENTORY
    # ======================================

    required_inventory = reorder_point


    # ======================================
    # REPLENISHMENT QUANTITY
    # ======================================

    replenishment_quantity = max(

        0,

        required_inventory -
        data.inventory_level

    )


    # ======================================
    # INVENTORY STATUS
    # ======================================

    if data.inventory_level <= safety_stock:

        inventory_status = "CRITICAL"

    elif data.inventory_level <= reorder_point:

        inventory_status = "REORDER"

    else:

        inventory_status = "SUFFICIENT"


    # ======================================
    # RETURN RESULTS
    # ======================================

    return {

        "predicted_demand":
            round(predicted_demand, 2),

        "lead_time_demand":
            round(lead_time_demand, 2),

        "safety_stock":
            round(safety_stock, 2),

        "reorder_point":
            round(reorder_point, 2),

        "required_inventory":
            round(required_inventory, 2),

        "replenishment_quantity":
            round(replenishment_quantity, 2),

        "inventory_status":
            inventory_status,

        "weather_condition":
            data.weather_condition

    }
