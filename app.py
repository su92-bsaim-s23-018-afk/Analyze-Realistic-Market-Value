from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

app = Flask(__name__)

BUNDLE_PATH = Path(__file__).resolve().parent / "advanced_pricing_bundle.pkl"
PIPELINE_PATH = Path(__file__).resolve().parent / "advanced_pricing_pipeline.pkl"

_bundle: dict[str, Any] | None = None

BRAND_PRESTIGE_MAP = {
    "Economy": 2,
    "Mid-Range": 5,
    "Luxury": 8,
    "Super-Car": 10,
}
VALID_FUEL_TYPES = {"Petrol", "Diesel", "Electric", "Hybrid"}
VALID_TRANSMISSIONS = {"Automatic", "Manual"}


def map_brand_band(value: int) -> str:
    if value <= 3:
        return "Economy"
    if value <= 6:
        return "Mid-Range"
    if value <= 8:
        return "Luxury"
    return "Super-Car"


def parse_accident_history(raw_value: Any) -> int:
    if isinstance(raw_value, str):
        value = raw_value.strip().replace("+", "")
    else:
        value = raw_value

    accidents = int(value)
    return max(0, min(accidents, 5))


def load_advanced_artifacts() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        if BUNDLE_PATH.exists():
            loaded = joblib.load(BUNDLE_PATH)
            if not isinstance(loaded, dict) or "pipeline" not in loaded:
                raise ValueError("advanced_pricing_bundle.pkl has an invalid format.")
            _bundle = loaded
        elif PIPELINE_PATH.exists():
            _bundle = {
                "pipeline": joblib.load(PIPELINE_PATH),
                "distance_scale": 1.0,
                "config_frequency": {},
            }
        else:
            raise FileNotFoundError(
                "Advanced model files are missing. Run train_advanced_agent.py first."
            )
    return _bundle


def compute_confidence_score(bundle: dict[str, Any], features: pd.DataFrame) -> tuple[float, float]:
    pipeline: Pipeline = bundle["pipeline"]
    distance_scale = float(bundle.get("distance_scale", 1.0))
    distance_scale = max(distance_scale, 1e-6)

    preprocessor = pipeline.named_steps.get("preprocessor")
    regressor = pipeline.named_steps.get("regressor")

    if preprocessor is None or not isinstance(regressor, KNeighborsRegressor):
        return 70.0, 0.0

    transformed = preprocessor.transform(features)
    distances = regressor.kneighbors(
        transformed,
        n_neighbors=regressor.n_neighbors,
        return_distance=True,
    )[0][0]
    mean_distance = float(np.mean(distances))

    confidence_score = 100.0 * np.exp(-(mean_distance / (distance_scale * 1.15)))
    confidence_score = float(np.clip(confidence_score, 8.0, 99.0))

    return confidence_score, mean_distance


def build_market_analysis(
    *,
    brand_prestige: int,
    fuel_type: str,
    transmission: str,
    accident_history: int,
    milage: int,
    confidence_score: float,
    config_frequency: dict[str, float],
) -> str:
    config_key = f"{fuel_type}|{transmission}|{map_brand_band(brand_prestige)}"
    frequency = float(config_frequency.get(config_key, 0.0))

    if frequency < 0.05:
        market_signal = "This car is rare in the current market."
    elif frequency > 0.14:
        market_signal = "This is a high-demand configuration in the current market."
    else:
        market_signal = "This configuration has balanced supply and demand in the current market."

    if confidence_score >= 85:
        confidence_signal = "Neighbor similarity is strong, so valuation confidence is high."
    elif confidence_score >= 65:
        confidence_signal = "Neighbor similarity is moderate, giving a stable estimate."
    else:
        confidence_signal = "Neighbor similarity is weaker, so market volatility may be higher."

    if accident_history == 0 and milage < 100_000:
        condition_signal = "Condition indicators support stronger buyer interest."
    elif accident_history >= 3:
        condition_signal = "Accident history may reduce buyer urgency and final offers."
    else:
        condition_signal = "Condition profile is in a typical range for this segment."

    return f"{market_signal} {confidence_signal} {condition_signal}"


def build_smart_message(
    *,
    year: int,
    milage: int,
    engine_size: float,
    fuel_type: str,
    predicted_price: float,
    confidence_score: float,
) -> str:
    vehicle_age = max(0, date.today().year - year)

    if predicted_price >= 65_000:
        segment_note = "premium valuation tier"
    elif predicted_price >= 35_000:
        segment_note = "mid-market valuation tier"
    else:
        segment_note = "value-driven valuation tier"

    if vehicle_age <= 4 and milage < 80_000:
        lifecycle_note = "modern lifecycle profile"
    else:
        lifecycle_note = "mature lifecycle profile"

    return (
        f"AI Agent Insight: Estimated in the {segment_note} with {confidence_score:.1f}% confidence. "
        f"{fuel_type} drivetrain, {engine_size:.1f}L engine, and {milage:,} mileage indicate a {lifecycle_note}."
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    required_fields = [
        "year",
        "milage",
        "engine_size",
        "brand_prestige",
        "fuel_type",
        "accident_history",
        "transmission",
    ]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        year = int(payload["year"])
        milage = int(payload["milage"])
        engine_size = float(payload["engine_size"])
        brand_tier = str(payload["brand_prestige"]).strip()
        fuel_type = str(payload["fuel_type"]).strip().title()
        transmission = str(payload["transmission"]).strip().title()
        accident_history = parse_accident_history(payload["accident_history"])
    except (TypeError, ValueError):
        return (
            jsonify(
                {
                    "error": "Invalid input types for one or more fields."
                }
            ),
            400,
        )

    if brand_tier not in BRAND_PRESTIGE_MAP:
        return (
            jsonify(
                {
                    "error": "brand_prestige must be one of: Economy, Mid-Range, Luxury, Super-Car."
                }
            ),
            400,
        )
    if fuel_type not in VALID_FUEL_TYPES:
        return jsonify({"error": "fuel_type must be Petrol, Diesel, Electric, or Hybrid."}), 400
    if transmission not in VALID_TRANSMISSIONS:
        return jsonify({"error": "transmission must be Automatic or Manual."}), 400

    current_year = date.today().year
    if year < 1980 or year > current_year + 1:
        return jsonify({"error": f"year must be between 1980 and {current_year + 1}."}), 400
    if milage < 0 or milage > 500_000:
        return jsonify({"error": "milage must be between 0 and 500000."}), 400
    if engine_size <= 0 or engine_size > 10:
        return jsonify({"error": "engine_size must be greater than 0 and less than or equal to 10."}), 400
    if accident_history < 0 or accident_history > 5:
        return jsonify({"error": "accident_history must be between 0 and 5."}), 400

    brand_prestige = BRAND_PRESTIGE_MAP[brand_tier]

    features = pd.DataFrame(
        [
            {
                "year": year,
                "milage": milage,
                "engine_size": engine_size,
                "brand_prestige": brand_prestige,
                "fuel_type": fuel_type,
                "accident_history": accident_history,
                "transmission": transmission,
            }
        ]
    )

    try:
        bundle = load_advanced_artifacts()
        pipeline: Pipeline = bundle["pipeline"]
        config_frequency = bundle.get("config_frequency", {})

        predicted_price = float(pipeline.predict(features)[0])
        confidence_score, _ = compute_confidence_score(bundle, features)
        market_analysis = build_market_analysis(
            brand_prestige=brand_prestige,
            fuel_type=fuel_type,
            transmission=transmission,
            accident_history=accident_history,
            milage=milage,
            confidence_score=confidence_score,
            config_frequency=config_frequency,
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception:
        return jsonify({"error": "Prediction failed due to an internal error."}), 500

    agent_message = build_smart_message(
        year=year,
        milage=milage,
        engine_size=engine_size,
        fuel_type=fuel_type,
        predicted_price=predicted_price,
        confidence_score=confidence_score,
    )

    return jsonify(
        {
            "predicted_price": round(predicted_price, 2),
            "currency": "USD",
            "confidence_score": round(confidence_score, 1),
            "market_analysis": market_analysis,
            "smart_agent_message": agent_message,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
