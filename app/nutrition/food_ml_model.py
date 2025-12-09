# app/nutrition/food_ml_model.py
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from joblib import dump, load

# Directory for food models (separate from alert models)
FOOD_MODELS_DIR = "models_food"
FOOD_META_PATH = os.path.join(FOOD_MODELS_DIR, "food_models_meta.json")

# -----------------------------
#  SIMPLE INDIAN BREAKFAST DB
# -----------------------------
# calories / protein / carbs / fat per "unit" (approx)
FOOD_DB: Dict[str, Dict[str, float]] = {
    "idli":           {"cal": 70,  "p": 2.0, "c": 12.0, "f": 0.5},
    "dosa":           {"cal": 130, "p": 3.0, "c": 20.0, "f": 4.0},
    "poha":           {"cal": 180, "p": 4.0, "c": 30.0, "f": 5.0},
    "upma":           {"cal": 200, "p": 5.0, "c": 28.0, "f": 7.0},
    "aloo_paratha":   {"cal": 220, "p": 5.0, "c": 30.0, "f": 9.0},
    "chai":           {"cal": 80,  "p": 2.0, "c": 10.0, "f": 3.0},
    "coffee":         {"cal": 60,  "p": 2.0, "c": 6.0,  "f": 2.0},
    "bread_slice":    {"cal": 75,  "p": 2.5, "c": 14.0, "f": 1.0},
    "omelette":       {"cal": 120, "p": 8.0, "c": 1.0,  "f": 9.0},
    "curd_bowl":      {"cal": 90,  "p": 5.0, "c": 4.0,  "f": 5.0},
    "paratha_plain":  {"cal": 190, "p": 4.0, "c": 28.0, "f": 7.0},
    "banana":         {"cal": 100, "p": 1.2, "c": 25.0, "f": 0.3},
}

FOOD_INDEX = {name: idx for idx, name in enumerate(FOOD_DB.keys())}


def _ensure_food_dirs():
    os.makedirs(FOOD_MODELS_DIR, exist_ok=True)
    if not os.path.exists(FOOD_META_PATH):
        with open(FOOD_META_PATH, "w") as f:
            json.dump({"models": []}, f, indent=2)


def _load_food_meta() -> Dict:
    _ensure_food_dirs()
    with open(FOOD_META_PATH, "r") as f:
        return json.load(f)


def _save_food_meta(meta: Dict):
    with open(FOOD_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def get_active_food_model_path() -> Optional[str]:
    meta = _load_food_meta()
    models = meta.get("models", [])
    active = [m for m in models if m.get("active")]
    if not active:
        return None
    return active[-1]["path"]


def get_registered_food_models() -> List[Dict]:
    meta = _load_food_meta()
    return meta.get("models", [])


# -----------------------------
#  TRAIN DATA GENERATION
# -----------------------------

def _generate_food_training_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Build synthetic training data from FOOD_DB.
    Features: [food_idx, quantity]
    Targets: [calories, protein, carbs, fat]
    """
    X = []
    y = []

    for name, idx in FOOD_INDEX.items():
        base = FOOD_DB[name]
        for qty in [0.5, 1.0, 1.5, 2.0, 3.0]:
            X.append([idx, qty])
            y.append([
                base["cal"] * qty,
                base["p"] * qty,
                base["c"] * qty,
                base["f"] * qty,
            ])

    return np.array(X), np.array(y)


# -----------------------------
#  MLOPS RETRAIN FOR FOOD MODEL
# -----------------------------

def retrain_food_model() -> Dict:
    """
    Train a MultiOutputRegressor on synthetic Indian breakfast data.
    Logged to MLflow & best model promoted via meta.
    """

    _ensure_food_dirs()

    X, y = _generate_food_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    mlflow.set_experiment("bodytwin_food_models")

    with mlflow.start_run(run_name="food_calorie_model") as run:
        base = RandomForestRegressor(n_estimators=200, random_state=42)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))

        mlflow.log_metric("mae", mae)
        mlflow.log_param("base_model", "RandomForestRegressor")
        mlflow.log_param("n_foods", len(FOOD_DB))

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(FOOD_MODELS_DIR, f"food_model_{timestamp}.joblib")
        dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path="food_model")

        meta = _load_food_meta()
        for m in meta.get("models", []):
            m["active"] = False

        meta.setdefault("models", []).append({
            "name": os.path.basename(model_path),
            "path": model_path,
            "mae": mae,
            "mlflow_run_id": run.info.run_id,
            "created_at": datetime.utcnow().isoformat(),
            "active": True,
        })
        _save_food_meta(meta)

        return {
            "status": "trained_food_model",
            "mae": mae,
            "model_path": model_path,
            "mlflow_run_id": run.info.run_id,
        }


# -----------------------------
#  PREDICTION HELPER
# -----------------------------

def _load_active_food_model():
    path = get_active_food_model_path()
    if not path or not os.path.exists(path):
        # If no model yet, train one once
        info = retrain_food_model()
        path = info["model_path"]
    return load(path)


def predict_meal_nutrition(items: List[Dict[str, float]]) -> Dict[str, float]:
    """
    items: [{ "name": "idli", "quantity": 2 }, ...]
    Returns summed calories, protein, carbs, fat.
    """
    model = _load_active_food_model()

    total_cal = 0.0
    total_p = 0.0
    total_c = 0.0
    total_f = 0.0

    for item in items:
        raw_name = item.get("name", "").strip().lower().replace(" ", "_")
        qty = float(item.get("quantity", 1.0))

        if raw_name not in FOOD_INDEX:
            # Unknown food → skip or assume 0; you can log warning
            continue

        idx = FOOD_INDEX[raw_name]
        X = np.array([[idx, qty]])
        cal, p, c, f = model.predict(X)[0]

        total_cal += cal
        total_p += p
        total_c += c
        total_f += f

    return {
        "calories": float(total_cal),
        "protein": float(total_p),
        "carbs": float(total_c),
        "fat": float(total_f),
    }
