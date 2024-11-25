from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import os
from IrisDataFilter import IrisDataFilter

# Initialize FastAPI app and IrisDataFilter instance
app = FastAPI()
data_filter = IrisDataFilter(dataset_path="iris.csv")


class FilterRequest(BaseModel):
    species: str
    features: Optional[List[str]] = None


@app.get("/")
def root():
    return {"message": "Welcome to the Iris Dataset API!"}


@app.post("/filter")
def filter_iris_data(request: FilterRequest):
    """Filter Iris data by species and visualize feature distribution."""
    species = request.species
    features = request.features or ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    
    try:
        # Filter data
        filtered_data = data_filter.filter_by_species(species)
        if filtered_data.empty:
            return {"error": f"No data found for species: {species}"}

        # Generate plot
        plot_file = data_filter.plot_feature_distribution(species, features)

        # Return results
        return {
            "filtered_data": filtered_data.to_dict(orient="records"),
            "visualization_file": plot_file
        }
    except Exception as e:
        return {"error": str(e)}
