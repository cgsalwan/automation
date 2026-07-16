
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app with expected variable name for gunicorn
superkart_api = Flask(__name__)

# Load the trained model
try:
    model = joblib.load("xgb_tuned_model.joblib")  # Make sure model is in the same directory
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Health check endpoint (optional but useful)
@superkart_api.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# Root welcome endpoint
@superkart_api.route("/", methods=["GET"])
def home():
    return "Welcome to SuperKart Sales Prediction API!"

# Prediction endpoint
@superkart_api.route("/v1/predict", methods=["POST"])
def predict_sales():
    try:
        data = request.get_json()

        required_features = [
            'Product_Weight',
            'Product_Sugar_Content',
            'Product_Allocated_Area',
            'Product_MRP',
            'Store_Size',
            'Store_Location_City_Type',
            'Store_Type',
            'Product_Id_char',
            'Store_Age_Years',
            'Product_Type_Category'
        ]

        # Ensure all required features are present
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing input features'}), 400

        # Create DataFrame for model input
        input_df = pd.DataFrame([data])

        # Predict
        prediction = model.predict(input_df)[0]

        return jsonify({"Sales": float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
