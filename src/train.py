"""
train.py
--------
Trains a Logistic Regression model with L2 regularization
to predict 30-day hospital readmission.

Includes:
- L2-regularized logistic regression (scikit-learn default penalty)
- Cross-validation to tune regularization strength (C)
- Comparison with L1 regularization (for discussion in report)
"""

import warnings
warnings.filterwarnings("ignore")  # suppress sklearn version-specific deprecation notices

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import joblib

from preprocess import load_data, clean_data, encode_features, split_and_scale


def train_l2_model(X_train, y_train):
    """
    Train Logistic Regression with L2 regularization.
    Use GridSearchCV to find the best regularization strength (C).
    Note: C is the INVERSE of regularization strength
    (smaller C = stronger regularization).
    """
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000)

    grid = GridSearchCV(
        model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"Best C (L2): {grid.best_params_['C']}")
    print(f"Best cross-val ROC-AUC: {grid.best_score_:.4f}")

    return grid.best_estimator_


def train_l1_model(X_train, y_train):
    """
    Optional: Train an L1-regularized model for comparison.
    L1 tends to push some coefficients to exactly zero (feature selection),
    while L2 shrinks all coefficients smoothly without zeroing them out.
    """
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)

    grid = GridSearchCV(
        model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"Best C (L1): {grid.best_params_['C']}")
    print(f"Best cross-val ROC-AUC: {grid.best_score_:.4f}")

    return grid.best_estimator_


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df_encoded = encode_features(df)

    X_train, X_test, y_train, y_test, feature_names = split_and_scale(df_encoded)

    print("=== Training L2-regularized Logistic Regression ===")
    l2_model = train_l2_model(X_train, y_train)
    joblib.dump(l2_model, "../data/l2_model.pkl")

    print("\n=== Training L1-regularized Logistic Regression (for comparison) ===")
    l1_model = train_l1_model(X_train, y_train)
    joblib.dump(l1_model, "../data/l1_model.pkl")

    # Save test set for evaluate.py to use
    np.save("../data/X_test.npy", X_test)
    np.save("../data/y_test.npy", y_test.to_numpy())

    print("\nModels trained and saved to data/ folder.")
    print("Run evaluate.py next to see performance metrics.")
