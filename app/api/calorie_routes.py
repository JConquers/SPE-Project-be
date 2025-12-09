# app/api/calorie_routes.py
from typing import List, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from sqlalchemy.orm import Session

from app.database.db import get_db
from app.nutrition.nutrition_service import (
    log_meal,
    get_daily_summary,
    get_meal_history,
)
from app.nutrition.nutrition_mlops import (
    retrain_food_mlops,
    list_food_models,
    active_food_model,
)

router = APIRouter()


# -----------------------------
#  Pydantic Schemas
# -----------------------------

class MealItem(BaseModel):
    name: str       # e.g. "idli", "aloo_paratha"
    quantity: float # in "units" as per FOOD_DB


class MealLogCreate(BaseModel):
    twin_id: int
    meal_type: str  # "breakfast" (for now)
    items: List[MealItem]


class MealLogRead(BaseModel):
    id: int
    twin_id: int
    date: str
    meal_type: str
    items: List[Dict[str, float]]
    calories: float
    protein: float
    carbs: float
    fat: float

    class Config:
        orm_mode = True


class DailySummaryRead(BaseModel):
    twin_id: int
    date: str
    total_calories: float
    required_calories: float
    calorie_balance: float

    class Config:
        orm_mode = True


# -----------------------------
#  ROUTES – MEAL TRACKING
# -----------------------------

@router.post("/meal", response_model=Dict[str, object])
def log_breakfast(
    payload: MealLogCreate,
    db: Session = get_db(),
):
    try:
        meal, daily = log_meal(
            db=db,
            twin_id=payload.twin_id,
            meal_type=payload.meal_type,
            items=[m.dict() for m in payload.items],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return {
        "meal": {
            "id": meal.id,
            "twin_id": meal.twin_id,
            "date": str(meal.date),
            "meal_type": meal.meal_type,
            "calories": meal.calories,
            "protein": meal.protein,
            "carbs": meal.carbs,
            "fat": meal.fat,
        },
        "daily_summary": {
            "twin_id": daily.twin_id,
            "date": str(daily.date),
            "total_calories": daily.total_calories,
            "required_calories": daily.required_calories,
            "calorie_balance": daily.calorie_balance,
        },
    }


@router.get("/daily/{twin_id}", response_model=DailySummaryRead)
def get_today_summary(
    twin_id: int,
    db: Session = get_db(),
):
    try:
        daily = get_daily_summary(db, twin_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return DailySummaryRead(
        twin_id=daily.twin_id,
        date=str(daily.date),
        total_calories=daily.total_calories,
        required_calories=daily.required_calories,
        calorie_balance=daily.calorie_balance,
    )


@router.get("/history/{twin_id}", response_model=List[MealLogRead])
def list_meals(
    twin_id: int,
    db: Session = get_db(),
):
    logs = get_meal_history(db, twin_id)
    result: List[MealLogRead] = []
    import json as _json

    for m in logs:
        try:
            items = _json.loads(m.food_json)
        except Exception:
            items = []
        result.append(
            MealLogRead(
                id=m.id,
                twin_id=m.twin_id,
                date=str(m.date),
                meal_type=m.meal_type,
                items=items,
                calories=m.calories,
                protein=m.protein,
                carbs=m.carbs,
                fat=m.fat,
            )
        )
    return result


# -----------------------------
#  ROUTES – FOOD MLOPS
# -----------------------------

@router.post("/mlops/retrain")
def retrain_food():
    """
    Retrain the food calorie model (separate from health alert model).
    """
    return retrain_food_mlops()


@router.get("/mlops/models")
def list_food_ml_models():
    """
    List all registered food models and their metrics.
    """
    return {"models": list_food_models()}


@router.get("/mlops/status")
def food_model_status():
    """
    Show active food model path (if any).
    """
    return active_food_model()
