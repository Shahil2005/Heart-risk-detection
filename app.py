from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import sys

app = Flask(__name__)

MODEL_PATH = 'models/models/heart_model.joblib'

# Load model on startup
model_data = None
try:
    if os.path.exists(MODEL_PATH):
        model_data = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Prediction endpoint will fail.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    if not model_data:
        return render_template('index.html', error="Model not loaded. Please train the model first.", columns=[])
    
    columns = model_data.get('columns', [])
    return render_template('index.html', columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_data:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate that all expected columns are present
        expected_columns = model_data['columns']
        missing_cols = [col for col in expected_columns if col not in data]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'}), 400
            
        # Create DataFrame for prediction
        # Ensure order matches training
        input_df = pd.DataFrame([data])[expected_columns]
        
        # Predict
        pipeline = model_data['model']
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
