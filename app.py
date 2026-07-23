import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'DecisionTree_model.pkl')
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)

# Categorical Feature Mappings
GENDER_MAP = {'Male': 0, 'Female': 1}
REGION_MAP = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Central': 4}
OCCUPATION_MAP = {'Professional': 0, 'Manager': 1, 'Clerk': 2, 'Student': 3, 'Other': 4}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Decision Tree Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --card-bg: rgba(17, 24, 39, 0.75);
            --border-glow: rgba(99, 102, 241, 0.2);
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --text-main: #f3f4f6;
            --text-sub: #9ca3af;
            --input-bg: #111827;
            --input-border: #374151;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 20% 20%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 80% 80%, rgba(168, 85, 247, 0.15) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }

        .card {
            width: 100%;
            max-width: 580px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-glow);
            border-radius: 20px;
            padding: 36px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .header p {
            color: var(--text-sub);
            font-size: 0.9rem;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }

        .full-width {
            grid-column: span 2;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-sub);
            margin-bottom: 6px;
        }

        input, select {
            background-color: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 10px;
            padding: 12px 14px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        .btn-submit {
            margin-top: 10px;
            padding: 14px;
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.1s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
        }

        .btn-submit:hover {
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        }

        .btn-submit:active {
            transform: scale(0.98);
        }

        #result-box {
            margin-top: 24px;
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            display: none;
            font-size: 1.1rem;
            font-weight: 600;
            animation: fadeIn 0.3s ease;
        }

        .result-yes {
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid #22c55e;
            color: #4ade80;
        }

        .result-no {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid #ef4444;
            color: #f87171;
        }

        .result-error {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid #dc2626;
            color: #fca5a5;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-6px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <div class="card">
        <div class="header">
            <h1>Decision Tree Predictor</h1>
            <p>Fill in the parameters below to evaluate model outcome</p>
        </div>

        <form id="prediction-form">
            <div class="grid">
                <div class="form-group">
                    <label>Age</label>
                    <input type="number" name="Age" value="35" min="18" max="100" required>
                </div>

                <div class="form-group">
                    <label>Gender</label>
                    <select name="Gender">
                        <option value="Male" selected>Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Region</label>
                    <select name="Region">
                        <option value="North">North</option>
                        <option value="South" selected>South</option>
                        <option value="East">East</option>
                        <option value="West">West</option>
                        <option value="Central">Central</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Occupation</label>
                    <select name="Occupation">
                        <option value="Professional" selected>Professional</option>
                        <option value="Manager">Manager</option>
                        <option value="Clerk">Clerk</option>
                        <option value="Student">Student</option>
                        <option value="Other">Other</option>
                    </select>
                </div>

                <div class="form-group full-width">
                    <label>Income ($)</label>
                    <input type="number" name="Income" value="50000" step="500" required>
                </div>

                <button type="submit" class="btn-submit full-width">Evaluate Prediction</button>
            </div>
        </form>

        <div id="result-box"></div>
    </div>

    <script>
        document.getElementById('prediction-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const resultBox = document.getElementById('result-box');
            resultBox.style.display = 'none';

            const formData = new FormData(this);
            const payload = Object.fromEntries(formData.entries());

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                resultBox.style.display = 'block';

                if (response.ok) {
                    const isYes = String(data.prediction).toLowerCase() === 'yes';
                    resultBox.className = isYes ? 'result-yes' : 'result-no';
                    resultBox.innerHTML = `Prediction Outcome: <strong>${data.prediction.toUpperCase()}</strong>`;
                } else {
                    resultBox.className = 'result-error';
                    resultBox.innerHTML = `Error: ${data.error}`;
                }
            } catch (err) {
                resultBox.style.display = 'block';
                resultBox.className = 'result-error';
                resultBox.innerHTML = 'Failed to communicate with prediction API';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'DecisionTree_model.pkl file missing.'}), 500

    try:
        if request.is_json:
            data = request.get_json()
        elif request.form:
            data = request.form
        else:
            return jsonify({'error': 'Unsupported Content-Type format'}), 415

        age = float(data.get('Age', 30))
        gender = GENDER_MAP.get(data.get('Gender'), 0)
        region = REGION_MAP.get(data.get('Region'), 0)
        occupation = OCCUPATION_MAP.get(data.get('Occupation'), 0)
        income = float(data.get('Income', 0))

        features = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Region': region,
            'Occupation': occupation,
            'Income': income
        }])

        prediction = model.predict(features)[0]

        return jsonify({
            'prediction': str(prediction)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
