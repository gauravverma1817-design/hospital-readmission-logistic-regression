

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

RAW_PATH = "../data/raw.csv"
PROCESSED_PATH = "../data/processed.csv"


def load_data(path=RAW_PATH):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    # Handle any missing values defensively
    df = df.dropna()

    # Remove obvious outliers/invalid entries (defensive bounds)
    df = df[(df["age"] >= 0) & (df["age"] <= 120)]
    df = df[(df["systolic_bp"] > 0)]

    return df


def encode_features(df):
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(
        df,
        columns=["gender", "diagnosis_code"],
        drop_first=True  # avoid dummy variable trap
    )
    return df_encoded


def split_and_scale(df, target_col="readmitted_30_days", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for later use (e.g. in evaluate.py or deployment)
    joblib.dump(scaler, "../data/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df_encoded = encode_features(df)

    # Save the cleaned + encoded dataset (before scaling) for inspection
    df_encoded.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_PATH}")
    print(f"Shape: {df_encoded.shape}")
    print(df_encoded.head())
