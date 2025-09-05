import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from joblib import load
from sqlalchemy import create_engine, text
from datetime import datetime

# -----------------------------
# Database config
# -----------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://admin:manubolusivananda@database-1.c32awucyihk7.us-west-1.rds.amazonaws.com:3306/paddy_disease_db"
)
engine = create_engine(DATABASE_URL, future=True)

# -----------------------------
# Load ML model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model.pkl")
model = load(MODEL_PATH)

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__, static_folder="/var/www/html", static_url_path="")
CORS(app)

# -----------------------------
# Health check
# -----------------------------
@app.get("/api/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "details": str(e)}), 500

# -----------------------------
# Predict API (Updated with debug + force JSON)
# -----------------------------
@app.post("/api/predict")
def predict():
    try:
        # ✅ Force JSON parsing and debug log
        data = request.get_json(force=True)
        print("📥 Incoming payload:", data)

        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        # Model expects only 7 numeric features
        features = [
            float(data.get("temperature", 28)),
            float(data.get("humidity", 80)),
            float(data.get("rainfall", 20)),
            float(data.get("nitrogen", 80)),
            float(data.get("phosphorus", 30)),
            float(data.get("potassium", 25)),
            float(data.get("leaf_wetness", 0.6))
        ]

        # Predict disease
        pred_label = model.predict([features])[0]

        # Save all data to MySQL
        sql = text("""
            INSERT INTO predictions
            (ts, farmer_name, location, temperature, humidity, rainfall,
             nitrogen, phosphorus, potassium, leaf_wetness, soil_moisture,
             image_url, predicted_disease)
            VALUES (:ts, :farmer_name, :location, :temperature, :humidity, :rainfall,
                    :nitrogen, :phosphorus, :potassium, :leaf_wetness, :soil_moisture,
                    :image_url, :predicted_disease)
        """)
        with engine.begin() as conn:
            conn.execute(sql, {
                "ts": datetime.utcnow(),
                "farmer_name": data.get("farmer_name", ""),
                "location": data.get("location", ""),
                "temperature": features[0],
                "humidity": features[1],
                "rainfall": features[2],
                "nitrogen": features[3],
                "phosphorus": features[4],
                "potassium": features[5],
                "leaf_wetness": features[6],
                "soil_moisture": float(data.get("soil_moisture", 0)),
                "image_url": data.get("image_url", ""),
                "predicted_disease": str(pred_label)
            })

        return jsonify({"prediction": str(pred_label)}), 200

    except Exception as e:
        print("❌ Error in /predict:", str(e))   # ✅ Log error
        return jsonify({"error": str(e)}), 400

# -----------------------------
# Fetch recent records
# -----------------------------
@app.get("/api/records")
def get_records():
    try:
        limit = int(request.args.get("limit", 100))
        limit = max(1, min(limit, 500))

        sql = text("""
            SELECT id, ts, farmer_name, location, temperature, humidity, rainfall,
                   nitrogen, phosphorus, potassium, leaf_wetness, soil_moisture,
                   image_url, predicted_disease
            FROM predictions
            ORDER BY id DESC
            LIMIT :limit
        """)
        with engine.connect() as conn:
            result = conn.execute(sql, {"limit": limit})
            rows = [dict(r._mapping) for r in result]

        return jsonify(rows), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------
# Serve frontend
# -----------------------------
@app.get("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

