"""Predict median house values for new districts with the saved pipeline.

Usage:  python src/predict.py data/input.csv [-o predictions.csv]
The CSV needs the same columns as housing.csv (median_house_value is optional and ignored).
"""
import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import features  # noqa: E402,F401  (needed to unpickle the custom transformer)
from features import TARGET  # noqa: E402

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "housing_model.joblib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--output", default="predictions.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    model = joblib.load(MODEL)
    df["predicted_median_house_value"] = model.predict(df.drop(columns=TARGET, errors="ignore")).round(-2)
    df.to_csv(args.output, index=False)
    print(df[["longitude", "latitude", "median_income", "ocean_proximity", "predicted_median_house_value"]]
          .to_string(index=False))
    print(f"\nSaved {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
