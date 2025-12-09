# app/nutrition/nutrition_mlops.py
from typing import Dict, List

from app.nutrition.food_ml_model import (
    retrain_food_model,
    get_registered_food_models,
    get_active_food_model_path,
)


def retrain_food_mlops() -> Dict:
    return retrain_food_model()


def list_food_models() -> List[Dict]:
    return get_registered_food_models()


def active_food_model() -> Dict:
    path = get_active_food_model_path()
    return {"active_food_model": path}
