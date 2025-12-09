# app/api/user_routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from app.database.db import get_db
from app.database.user_schema import User, UserCreate, UserRead

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@router.post("/register", response_model=UserRead)
def register_user(payload: UserCreate, db: Session = get_db()):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # hashed = pwd_context.hash(payload.password)
    raw_password = payload.password.strip()

    if len(raw_password.encode("utf-8")) > 72:
        raw_password = raw_password[:72]

    hashed = pwd_context.hash(raw_password)
    user = User(email=payload.email, name=payload.name, password_hash=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get("/{user_id}", response_model=UserRead)
def get_user(user_id: int, db: Session = get_db()):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
