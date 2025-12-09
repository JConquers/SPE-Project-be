# app/nutrition/nutrition_models.py
from datetime import datetime, date
from sqlalchemy import Column, Integer, Float, String, Date, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.database.db import Base


class MealLog(Base):
    __tablename__ = "meal_logs"

    id = Column(Integer, primary_key=True, index=True)
    twin_id = Column(Integer, ForeignKey("twins.id"), index=True, nullable=False)

    date = Column(Date, default=date.today, index=True)
    meal_type = Column(String, nullable=False)  # "breakfast" (for now)

    # JSON string of items: [{"name": "idli", "quantity": 2}, ...]
    food_json = Column(Text, nullable=False)

    calories = Column(Float, nullable=False)
    protein = Column(Float, nullable=False)
    carbs = Column(Float, nullable=False)
    fat = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    twin = relationship("Twin")


class DailyCalorieSummary(Base):
    __tablename__ = "daily_calorie_summary"

    id = Column(Integer, primary_key=True, index=True)
    twin_id = Column(Integer, ForeignKey("twins.id"), index=True, nullable=False)
    date = Column(Date, default=date.today, index=True)

    total_calories = Column(Float, default=0.0)
    required_calories = Column(Float, default=0.0)
    calorie_balance = Column(Float, default=0.0)  # total - required

    twin = relationship("Twin")
