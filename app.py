"""
House Price Estimator - Flask backend
--------------------------------------
Loads a pre-trained scikit-learn LinearRegression model (linear_model.pkl)
and serves a small web UI + JSON prediction endpoint.

Deploy target: Render (Web Service)
Local run:      python app.py
Production run: gunicorn app:app
"""

import os
import warnings
import pickle

import numpy as np
from flask import Flask, request, jsonify, render_template

# Silence the harmless "InconsistentVersionWarning" that scikit-learn raises
# when the model was pickled with a slightly different sklearn version than
# the one installed at load time. It does not affect .predict() results.
warnings.filterwarnings("ignore")

app = Flask(__name__)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "linear_model.pkl")

# Change this if your model's target is in a different currency.
CURRENCY_SYMBOL = "$"

# The exact order of features the model was trained on. Do NOT reorder --
# a LinearRegression model has no idea what a feature "means", it only
# knows position, so this list must match the training column order.
FEATURE_ORDER = [
    "bedrooms",
    "bathrooms",
    "living_area",
    "lot_area",
    "floors",
    "waterfront",
    "views",
    "condition",
    "grade",
    "house_area_excl_basement",
    "basement_area",
    "built_year",
    "renovation_year",
    "lot_area_renov",
    "schools_nearby",
    "airport_distance",
]

# Human-friendly labels shown in the UI, mapped 1:1 to FEATURE_ORDER.
FIELD_LABELS = {
    "bedrooms": "Number of Bedrooms",
    "bathrooms": "Number of Bathrooms",
    "living_area": "Living Area (sqft)",
    "lot_area": "Lot Area (sqft)",
    "floors": "Number of Floors",
    "waterfront": "Waterfront Present",
    "views": "Number of Views",
    "condition": "Condition of the House",
    "grade": "Grade of the House",
    "house_area_excl_basement": "House Area Excl. Basement (sqft)",
    "basement_area": "Basement Area (sqft)",
    "built_year": "Built Year",
    "renovation_year": "Renovation Year",
    "lot_area_renov": "Renovated Lot Area (sqft)",
    "schools_nearby": "Schools Nearby",
    "airport_distance": "Distance from Airport (km)",
}

# --------------------------------------------------------------------------
# Load model once at startup
# --------------------------------------------------------------------------
model = None
model_load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as exc:  # noqa: BLE001 - we want to report ANY load failure
    model_load_error = str(exc)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    """Render the main UI."""
    return render_template("index.html", currency=CURRENCY_SYMBOL)


@app.route("/health")
def health():
    """Simple health-check endpoint (handy for Render)."""
    status = "ok" if model is not None else "model_not_loaded"
    return jsonify({"status": status}), (200 if model is not None else 500)


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a JSON payload of house features and return a price estimate."""
    if model is None:
        return jsonify({
            "success": False,
            "error": f"Model failed to load on the server: {model_load_error}",
        }), 500

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"success": False, "error": "No input data received."}), 400

    values = []
    for key in FEATURE_ORDER:
        raw_value = payload.get(key, "")
        if raw_value in ("", None):
            label = FIELD_LABELS.get(key, key)
            return jsonify({
                "success": False,
                "error": f"'{label}' is required.",
            }), 400
        try:
            values.append(float(raw_value))
        except (TypeError, ValueError):
            label = FIELD_LABELS.get(key, key)
            return jsonify({
                "success": False,
                "error": f"'{label}' must be a number.",
            }), 400

    try:
        features = np.array(values, dtype=float).reshape(1, -1)
        raw_prediction = model.predict(features)[0]
        prediction = round(max(float(raw_prediction), 0), 2)
        return jsonify({"success": True, "prediction": prediction})
    except Exception as exc:  # noqa: BLE001
        return jsonify({
            "success": False,
            "error": f"The model could not generate a prediction: {exc}",
        }), 500


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"success": False, "error": "Route not found."}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"success": False, "error": "Internal server error."}), 500


# --------------------------------------------------------------------------
# Local dev entrypoint (Render uses gunicorn, defined in the Procfile)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
