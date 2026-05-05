from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
DATASET_SIZE = 100_000
DATASET_FILENAME = "synthetic_car_dataset.csv"


def generate_synthetic_data(rows: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    current_year = 2026

    years = rng.integers(2000, current_year + 1, size=rows)
    ages = current_year - years

    annual_km = rng.uniform(8_000, 20_000, size=rows)
    mileage_noise = rng.normal(0, 12_000, size=rows)
    milage = np.clip((ages * annual_km) + mileage_noise, 0, 320_000).round().astype(int)

    engine_size = rng.uniform(1.0, 5.2, size=rows).round(1)

    base_price = 44_000 + (years - 2000) * 1_150
    mileage_penalty = milage * 0.055
    engine_bonus = engine_size * 3_700
    condition_boost = np.where(milage < 55_000, 2_400, 0)
    premium_engine_boost = np.where(engine_size >= 3.5, 3_200, 0)
    random_noise = rng.normal(0, 2_900, size=rows)

    price = (
        base_price
        - mileage_penalty
        + engine_bonus
        + condition_boost
        + premium_engine_boost
        + random_noise
    )
    price = np.clip(price, 2_500, 120_000).round(2)

    return pd.DataFrame(
        {
            "year": years.astype(int),
            "milage": milage,
            "engine_size": engine_size,
            "price": price,
        }
    )


def train_and_save_model() -> None:
    df = generate_synthetic_data(DATASET_SIZE)
    root = Path(__file__).resolve().parent
    dataset_path = root / DATASET_FILENAME

    # Persist the generated synthetic dataset for inspection and reuse.
    df.to_csv(dataset_path, index=False)

    X = df[["year", "milage", "engine_size"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsRegressor(n_neighbors=9, weights="distance", metric="minkowski", p=2)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    joblib.dump(model, root / "car_model.pkl")
    joblib.dump(scaler, root / "scaler.pkl")

    print("Training finished.")
    print(f"Rows generated: {len(df)}")
    print(f"MAE: {mae:,.2f}")
    print(f"R2 score: {r2:.4f}")
    print(f"Saved dataset to: {dataset_path}")
    print(f"Saved model to: {root / 'car_model.pkl'}")
    print(f"Saved scaler to: {root / 'scaler.pkl'}")


if __name__ == "__main__":
    train_and_save_model()
