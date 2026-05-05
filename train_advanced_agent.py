from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
DATASET_SIZE = 50_000
CURRENT_YEAR = 2026

FUEL_TYPES = ["Petrol", "Diesel", "Electric", "Hybrid"]
TRANSMISSIONS = ["Automatic", "Manual"]


def generate_synthetic_data(rows: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    years = rng.integers(2000, CURRENT_YEAR + 1, size=rows)
    ages = CURRENT_YEAR - years

    annual_km = rng.uniform(9_000, 22_000, size=rows)
    mileage_noise = rng.normal(0, 11_000, size=rows)
    milage = np.clip((ages * annual_km) + mileage_noise, 0, 380_000).round().astype(int)

    engine_size = rng.uniform(1.0, 5.8, size=rows).round(1)

    prestige_distribution = np.array([0.14, 0.16, 0.16, 0.14, 0.12, 0.1, 0.07, 0.05, 0.04, 0.02])
    brand_prestige = rng.choice(np.arange(1, 11), size=rows, p=prestige_distribution)

    fuel_type = []
    for year in years:
        if year >= 2020:
            probs = [0.38, 0.18, 0.22, 0.22]
        elif year >= 2014:
            probs = [0.45, 0.28, 0.08, 0.19]
        else:
            probs = [0.52, 0.36, 0.02, 0.1]
        fuel_type.append(rng.choice(FUEL_TYPES, p=probs))
    fuel_type = np.array(fuel_type)

    auto_bias = (
        0.32
        + (brand_prestige * 0.045)
        + np.where(np.isin(fuel_type, ["Electric", "Hybrid"]), 0.16, 0)
    )
    auto_probability = np.clip(auto_bias, 0.2, 0.96)
    transmission = np.where(rng.random(rows) < auto_probability, "Automatic", "Manual")

    accident_signal = (milage / 120_000) + (ages / 8)
    accident_noise = rng.normal(0, 0.8, size=rows)
    accident_history = np.clip(np.rint(accident_signal + accident_noise), 0, 5).astype(int)

    fuel_effect = pd.Series(fuel_type).map(
        {
            "Petrol": 0,
            "Diesel": 2_000,
            "Hybrid": 4_800,
            "Electric": 8_800,
        }
    ).to_numpy()

    transmission_effect = np.where(transmission == "Automatic", 1_350, -450)

    base_price = 46_000 + (years - 2000) * 1_280
    mileage_penalty = milage * 0.058
    engine_bonus = engine_size * 3_950
    prestige_bonus = brand_prestige * 1_760
    accident_penalty = accident_history * 2_300
    age_penalty = (ages**1.12) * 250
    noise = rng.normal(0, 2_450, size=rows)

    price = (
        base_price
        - mileage_penalty
        + engine_bonus
        + prestige_bonus
        + fuel_effect
        + transmission_effect
        - accident_penalty
        - age_penalty
        + noise
    )
    price = np.clip(price, 3_200, 190_000).round(2)

    return pd.DataFrame(
        {
            "year": years.astype(int),
            "milage": milage,
            "engine_size": engine_size,
            "brand_prestige": brand_prestige.astype(int),
            "fuel_type": fuel_type,
            "accident_history": accident_history,
            "transmission": transmission,
            "price": price,
        }
    )


def map_brand_band(value: int) -> str:
    if value <= 3:
        return "Economy"
    if value <= 6:
        return "Mid-Range"
    if value <= 8:
        return "Luxury"
    return "Super-Car"


def train_and_save_advanced_model() -> None:
    df = generate_synthetic_data(DATASET_SIZE)

    features = [
        "year",
        "milage",
        "engine_size",
        "brand_prestige",
        "fuel_type",
        "accident_history",
        "transmission",
    ]
    numerical_features = ["year", "milage", "engine_size", "brand_prestige", "accident_history"]
    categorical_features = ["fuel_type", "transmission"]

    X = df[features]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                KNeighborsRegressor(
                    n_neighbors=25,
                    weights="distance",
                    metric="minkowski",
                    p=2,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    regressor = pipeline.named_steps["regressor"]
    transformed_calibration = pipeline.named_steps["preprocessor"].transform(
        X_test.sample(n=min(3_000, len(X_test)), random_state=RANDOM_SEED)
    )
    distances = regressor.kneighbors(transformed_calibration, n_neighbors=regressor.n_neighbors)[0]
    mean_distances = distances.mean(axis=1)
    distance_scale = float(np.percentile(mean_distances, 90))

    brand_band = df["brand_prestige"].map(map_brand_band)
    config_key = df["fuel_type"] + "|" + df["transmission"] + "|" + brand_band
    config_frequency = config_key.value_counts(normalize=True).to_dict()

    root = Path(__file__).resolve().parent
    pipeline_path = root / "advanced_pricing_pipeline.pkl"
    bundle_path = root / "advanced_pricing_bundle.pkl"

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(
        {
            "pipeline": pipeline,
            "distance_scale": distance_scale,
            "config_frequency": config_frequency,
            "feature_order": features,
            "metrics": {
                "mae": float(mae),
                "r2": float(r2),
            },
        },
        bundle_path,
    )

    print("Advanced training finished.")
    print(f"Rows generated: {len(df)}")
    print(f"MAE: {mae:,.2f}")
    print(f"R2 score: {r2:.4f}")
    print(f"Distance scale (confidence calibration): {distance_scale:.5f}")
    print(f"Saved full pipeline to: {pipeline_path}")
    print(f"Saved pipeline bundle to: {bundle_path}")


if __name__ == "__main__":
    train_and_save_advanced_model()
