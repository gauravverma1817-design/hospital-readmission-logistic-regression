import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)
import joblib


def load_test_data():
    X_test = np.load("../data/X_test.npy")
    y_test = np.load("../data/y_test.npy")
    return X_test, y_test


def evaluate_model(model, X_test, y_test, model_name="L2 Logistic Regression"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {model_name} Evaluation =====")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Readmit", "Readmit"]))

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Readmit", "Readmit"]).plot(ax=ax, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"../data/confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.close()

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    plt.title(f"ROC Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(f"../data/roc_curve_{model_name.replace(' ', '_')}.png")
    plt.close()

    return cm, auc


def discuss_clinical_cost(cm):
    """
    Prints a discussion of the clinical cost tradeoff between
    False Positives and False Negatives in readmission prediction.
    """
    tn, fp, fn, tp = cm.ravel()

    print("\n===== Clinical Cost Discussion =====")
    print(f"False Positives (predicted readmit, actually did NOT readmit): {fp}")
    print(f"False Negatives (predicted no readmit, actually DID readmit): {fn}")
    print("""
In hospital readmission prediction, a FALSE NEGATIVE is clinically far more
costly than a FALSE POSITIVE:

- A False Negative means a high-risk patient is sent home WITHOUT extra
  follow-up care, monitoring, or intervention -- increasing the chance of a
  preventable readmission, complications, or even patient harm.

- A False Positive means a lower-risk patient receives some extra follow-up
  (a phone call, an extra check-up) that turns out to be unnecessary --
  costly in staff time, but not dangerous to the patient.

Because of this asymmetry, hospitals typically prefer a model with HIGHER
RECALL (sensitivity) even if it costs some Precision -- i.e., it's better to
flag some patients unnecessarily than to miss patients who truly need
follow-up care. This can be achieved by LOWERING the classification
threshold below the default 0.5, at the cost of more false alarms.
""")


if __name__ == "__main__":
    X_test, y_test = load_test_data()

    l2_model = joblib.load("../data/l2_model.pkl")
    l1_model = joblib.load("../data/l1_model.pkl")

    cm_l2, auc_l2 = evaluate_model(l2_model, X_test, y_test, "L2 Logistic Regression")
    discuss_clinical_cost(cm_l2)

    cm_l1, auc_l1 = evaluate_model(l1_model, X_test, y_test, "L1 Logistic Regression")

    print("\n===== L1 vs L2 Comparison =====")
    print(f"L2 ROC-AUC: {auc_l2:.4f}")
    print(f"L1 ROC-AUC: {auc_l1:.4f}")
    print("L1 tends to zero-out less useful features (built-in feature selection).")
    print("L2 shrinks all coefficients smoothly, keeping every feature slightly active.")
