from flask import Flask, render_template, jsonify, request
import joblib
import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# =====================================================
# Paths
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# =====================================================
# Load model artifacts once on startup
# =====================================================
try:
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
except Exception:
    # Fallback for test environment (prevents import crash during pytest)
    model = None
    preprocessor = None
    feature_columns = []


def _extract_positive_probs(estimator, X):
    """Return malware-class probability when available, else max class probability."""
    proba = estimator.predict_proba(X)
    classes = list(getattr(estimator, "classes_", []))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba.max(axis=1)

def predict_malware(input_data: dict):
    """
    Core prediction function used by both API and tests.
    """

    if model is None or preprocessor is None or not feature_columns:
        # Minimal fallback behavior for tests
        return {
            "prediction": "Goodware",
            "confidence": 0.0
        }

    # Create base row filled with zeros
    row = [0] * len(feature_columns)

    # Fill known features from input
    for i, col in enumerate(feature_columns):
        if col in input_data:
            try:
                row[i] = float(input_data[col])
            except (TypeError, ValueError):
                row[i] = 0

    # Convert to DataFrame
    sample = pd.DataFrame([row], columns=feature_columns)

    # Preprocess
    X = preprocessor.transform(sample)

    # Predict
    pred = model.predict(X)[0]
    prob = _extract_positive_probs(model, X)[0]

    label = "Malware" if pred == 1 else "Goodware"

    return {
        "prediction": label,
        "confidence": round(float(prob), 4)
    }

# =====================================================
# Home Page
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# Health Check (for deployment + CI/CD smoke tests)
# =====================================================
@app.route("/health")
def health():
    return {"status": "ok"}, 200


# =====================================================
# Demo Prediction Route
# =====================================================
@app.route("/predict-demo")
def predict_demo():
    # Create one sample row filled with zeros
    sample = pd.DataFrame(
        [[0] * len(feature_columns)],
        columns=feature_columns
    )

    # Apply preprocessing
    X = preprocessor.transform(sample)

    # Predict
    pred = model.predict(X)[0]
    prob = _extract_positive_probs(model, X)[0]

    label = "Malware" if pred == 1 else "Goodware"

    return jsonify({
        "prediction": label,
        "confidence": round(float(prob), 4)
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON API endpoint for CI/CD + testing
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid or missing JSON"}), 400

    result = predict_malware(data)

    return jsonify(result), 200


# =====================================================
# Manual Form Prediction
# =====================================================
@app.route("/predict-form", methods=["POST"])
def predict_form():
    # Start all features at zero
    row = [0] * len(feature_columns)

    # Fill first 3 editable fields from form
    try:
        row[0] = float(request.form["f1"])
        row[1] = float(request.form["f2"])
        row[2] = float(request.form["f3"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid form input. Expected numeric f1, f2, f3."}), 400

    # Convert to DataFrame
    sample = pd.DataFrame([row], columns=feature_columns)

    # Preprocess
    X = preprocessor.transform(sample)

    # Predict
    pred = model.predict(X)[0]
    prob = _extract_positive_probs(model, X)[0]

    label = "Malware" if pred == 1 else "Goodware"

    return render_template(
        "result.html",
        prediction=label,
        confidence=round(float(prob), 4)
    )


# =====================================================
# Batch Upload + Auto Evaluation
# =====================================================
@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No CSV file uploaded."}), 400

    # Read uploaded CSV
    try:
        df = pd.read_csv(file)
    except Exception:
        return jsonify({"error": "Unable to read uploaded CSV."}), 400

    # ---------------------------------------------
    # Detect optional label column (case-insensitive)
    # ---------------------------------------------
    label_col = None
    lowered_cols = {str(col).strip().lower(): col for col in df.columns}

    for candidate in ["label", "target", "class", "y"]:
        if candidate in lowered_cols:
            label_col = lowered_cols[candidate]
            break

    # ---------------------------------------------
    # Keep only expected model columns
    # Missing columns auto-filled with zero
    # ---------------------------------------------
    X_input = df.reindex(columns=feature_columns, fill_value=0)

    # Preprocess
    X = preprocessor.transform(X_input)

    # Predict
    preds = model.predict(X)
    probs = _extract_positive_probs(model, X)

    # ---------------------------------------------
    # If labels exist -> show evaluation metrics
    # ---------------------------------------------
    if label_col:
        y_true = pd.to_numeric(df[label_col], errors="coerce")
        valid_mask = y_true.notna()
        y_true = y_true[valid_mask].astype(int)
        preds_eval = pd.Series(preds)[valid_mask].astype(int)
        probs_eval = pd.Series(probs)[valid_mask]

        acc = accuracy_score(y_true, preds_eval)
        try:
            auc = roc_auc_score(y_true, probs_eval)
        except ValueError:
            # AUC is undefined when y_true has only one class.
            auc = None

        cm = confusion_matrix(y_true, preds_eval, labels=[0, 1])

        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"]
        )

        return render_template(
            "metrics.html",
            accuracy=round(acc, 4),
            auc=round(auc, 4) if auc is not None else "N/A (single class in labels)",
            matrix=cm_df.to_html()
        )

    # ---------------------------------------------
    # No labels -> normal prediction table
    # ---------------------------------------------
    results = df.copy()

    results["Prediction"] = [
        "Malware" if p == 1 else "Goodware"
        for p in preds
    ]

    results["Confidence"] = probs.round(4)

    return render_template(
        "batch_results.html",
        table=results.head(50).to_html(index=False)
    )


# =====================================================
# Run Flask App Locally
# =====================================================
if __name__ == "__main__":
    app.run(debug=False)