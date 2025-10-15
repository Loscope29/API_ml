from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Charger le modèle une fois au démarrage
model = None
threshold = None

def load_model():
    global model, threshold
    try:
        # Chemin absolu pour Vercel
        model_path = os.path.join(os.path.dirname(__file__), '..', 'rf_optimized.pkl')
        data = joblib.load(model_path)
        model = data['model']
        threshold = data['threshold']
        print("✅ Modèle chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")

# Charger au démarrage
load_model()

@app.route('/')
def home():
    return jsonify({
        "message": "API de prédiction ML", 
        "status": "active",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)

        pred_proba = model.predict_proba(features)[:, 1]
        prediction = int(pred_proba[0] >= threshold)

        return jsonify({
            "prediction": prediction,
            "probability": float(pred_proba[0]),
            "threshold": float(threshold)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

# Nécessaire pour Vercel
app = app