from fastapi import FastAPI
from pydantic import BaseModel

from model import HousePricePredictor

app = FastAPI()

# Load the model
model = HousePricePredictor('data/house_prices.csv')
model.load_data()
model.preprocess_data()
model.train_model()

# Define the input data structure
class HouseFeatures(BaseModel):
    Id: float
    SalesPrice: float
  

# Create the prediction endpoint
@app.post("/predict_price")
def predict_price(features: HouseFeatures):
    predicted_price = model.predict_price([features.feature1, features.feature2])
    return {"predicted_price": predicted_price}