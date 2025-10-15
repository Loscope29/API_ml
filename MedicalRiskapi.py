from flask import Flask, request, jsonify
import joblib
import numpy as np

# Charger le dictionnaire
data = joblib.load("rf_optimized.pkl")
model = data['model']
threshold = data['threshold']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)

        pred_proba = model.predict_proba(features)[:, 1]
        prediction = int(pred_proba[0] >= threshold)

        return jsonify({"prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
