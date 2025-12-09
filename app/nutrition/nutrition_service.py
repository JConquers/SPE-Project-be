# app/nutrition/nutrition_service.py
from datetime import date
import json
from typing import List, Dict, Tuple

from sqlalchemy.orm import Session

from app.database.twin_schema import Twin
from app.nutrition.nutrition_models import MealLog, DailyCalorieSummary
from app.nutrition.food_ml_model import predict_meal_nutrition


def _estimate_required_calories(twin: Twin) -> float:
    """
    Simple BMR-based daily calorie need approximation.
    You can explain this formula in your report.
    """
    if not twin.height_cm or not twin.weight_kg or not twin.age:
        return 2000.0

    height = twin.height_cm
    weight = twin.weight_kg
    age = twin.age

    # Mifflin-St Jeor-like approximation
    if twin.gender.lower().startswith("m"):
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # Activity factor based on exercise_level (0-3)
    level = twin.exercise_level or 1
    if level == 0:
        factor = 1.2
    elif level == 1:
        factor = 1.375
    elif level == 2:
        factor = 1.55
    else:
        factor = 1.725

    return float(bmr * factor)


def log_meal(
    db: Session,
    twin_id: int,
    meal_type: str,
    items: List[Dict[str, float]],
) -> Tuple[MealLog, DailyCalorieSummary]:
    """
    Core function:
    - uses ML to predict meal nutrition
    - logs MealLog
    - updates DailyCalorieSummary
    """
    twin = db.query(Twin).filter(Twin.id == twin_id).first()
    if not twin:
        raise ValueError("Twin not found")

    nutrition = predict_meal_nutrition(items)

    today = date.today()

    meal = MealLog(
        twin_id=twin_id,
        date=today,
        meal_type=meal_type,
        food_json=json.dumps(items),
        calories=nutrition["calories"],
        protein=nutrition["protein"],
        carbs=nutrition["carbs"],
        fat=nutrition["fat"],
    )
    db.add(meal)

    daily = (
        db.query(DailyCalorieSummary)
        .filter(
            DailyCalorieSummary.twin_id == twin_id,
            DailyCalorieSummary.date == today,
        )
        .first()
    )

    if not daily:
        required = _estimate_required_calories(twin)
        daily = DailyCalorieSummary(
            twin_id=twin_id,
            date=today,
            total_calories=nutrition["calories"],
            required_calories=required,
        )
        daily.calorie_balance = daily.total_calories - daily.required_calories
        db.add(daily)
    else:
        daily.total_calories += nutrition["calories"]
        daily.calorie_balance = daily.total_calories - daily.required_calories

    db.commit()
    db.refresh(meal)
    db.refresh(daily)

    return meal, daily


def get_daily_summary(db: Session, twin_id: int) -> DailyCalorieSummary:
    today = date.today()
    daily = (
        db.query(DailyCalorieSummary)
        .filter(
            DailyCalorieSummary.twin_id == twin_id,
            DailyCalorieSummary.date == today,
        )
        .first()
    )

    if daily:
        return daily

    # If no record yet, compute baseline with 0 consumed
    twin = db.query(Twin).filter(Twin.id == twin_id).first()
    if not twin:
        raise ValueError("Twin not found")

    required = _estimate_required_calories(twin)
    daily = DailyCalorieSummary(
        twin_id=twin_id,
        date=today,
        total_calories=0.0,
        required_calories=required,
        calorie_balance=-required,
    )
    db.add(daily)
    db.commit()
    db.refresh(daily)
    return daily


def get_meal_history(db: Session, twin_id: int) -> List[MealLog]:
    return (
        db.query(MealLog)
        .filter(MealLog.twin_id == twin_id)
        .order_by(MealLog.date.desc(), MealLog.created_at.desc())
        .all()
    )
