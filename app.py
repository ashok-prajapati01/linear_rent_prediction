import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Feature names as expected by the trained Scikit-Learn model
FEATURE_NAMES = [
    "number of bedrooms",
    "number of bathrooms",
    "living area",
    "lot area",
    "number of floors",
    "waterfront present",
    "number of views",
    "condition of the house",
    "grade of the house",
    "Area of the house(excluding basement)",
    "Area of the basement",
    "Built Year",
    "Renovation Year",
    "lot_area_renov",
    "Number of schools nearby",
    "Distance from the airport"
]

# Field metadata for generating beautiful UI controls
FEATURE_CONFIG = [
    {"name": "number of bedrooms", "label": "Bedrooms", "type": "number", "default": 3, "min": 0, "max": 10, "step": 1},
    {"name": "number of bathrooms", "label": "Bathrooms", "type": "number", "default": 2, "min": 0, "max": 10, "step": 0.5},
    {"name": "living area", "label": "Living Area (sq ft)", "type": "number", "default": 2000, "min": 100, "step": 10},
    {"name": "lot area", "label": "Lot Area (sq ft)", "type": "number", "default": 5000, "min": 100, "step": 10},
    {"name": "number of floors", "label": "Floors", "type": "number", "default": 1, "min": 1, "max": 4, "step": 0.5},
    {"name": "waterfront present", "label": "Waterfront", "type": "select", "options": [{"val": 0, "label": "No"}, {"val": 1, "label": "Yes"}]},
    {"name": "number of views", "label": "View Rating (0-4)", "type": "number", "default": 0, "min": 0, "max": 4, "step": 1},
    {"name": "condition of the house", "label": "Condition (1-5)", "type": "number", "default": 3, "min": 1, "max": 5, "step": 1},
    {"name": "grade of the house", "label": "Grade (1-13)", "type": "number", "default": 7, "min": 1, "max": 13, "step": 1},
    {"name": "Area of the house(excluding basement)", "label": "Above Ground Area (sq ft)", "type": "number", "default": 1500, "min": 100, "step": 10},
    {"name": "Area of the basement", "label": "Basement Area (sq ft)", "type": "number", "default": 500, "min": 0, "step": 10},
    {"name": "Built Year", "label": "Year Built", "type": "number", "default": 2000, "min": 1800, "max": 2026, "step": 1},
    {"name": "Renovation Year", "label": "Renovation Year (0 if none)", "type": "number", "default": 0, "min": 0, "max": 2026, "step": 1},
    {"name": "lot_area_renov", "label": "Renovated Lot Area", "type": "number", "default": 5000, "min": 0, "step": 10},
    {"name": "Number of schools nearby", "label": "Nearby Schools", "type": "number", "default": 2, "min": 0, "max": 10, "step": 1},
    {"name": "Distance from the airport", "label": "Airport Distance (km)", "type": "number", "default": 15, "min": 0, "step": 0.1}
]

# Load model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")

# HTML/CSS UI with Glassmorphic effects and Neon standard styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price AI Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(18, 24, 38, 0.7);
            --border-glow: rgba(99, 102, 241, 0.3);
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --input-bg: rgba(30, 41, 59, 0.6);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            position: relative;
            overflow-x: hidden;
        }

        /* Unreal Glowing Background Orbs */
        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(120px);
            z-index: 0;
            pointer-events: none;
            opacity: 0.5;
        }
        .orb-1 { width: 400px; height: 400px; background: #6366f1; top: -50px; left: -50px; }
        .orb-2 { width: 350px; height: 350px; background: #ec4899; bottom: -50px; right: -50px; }

        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 950px;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-glow);
            border-radius: 28px;
            padding: 2.5rem;
            box-shadow: 
                0 0 50px rgba(99, 102, 241, 0.15),
                0 20px 40px rgba(0, 0, 0, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .header p {
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-size: 0.95rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.82rem;
            font-weight: 600;
            color: #d1d5db;
            letter-spacing: 0.3px;
        }

        .input-control {
            background: var(--input-bg);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        }

        .input-control:focus {
            border-color: #a855f7;
            box-shadow: 
                0 0 0 3px rgba(168, 85, 247, 0.25),
                0 0 15px rgba(168, 85, 247, 0.3);
        }

        select.input-control {
            cursor: pointer;
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: var(--accent-gradient);
            border: none;
            border-radius: 14px;
            padding: 1rem;
            color: white;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 
                0 10px 25px -5px rgba(168, 85, 247, 0.4),
                0 0 20px rgba(236, 72, 153, 0.3);
        }

        .btn-submit:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 
                0 15px 35px -5px rgba(168, 85, 247, 0.6),
                0 0 30px rgba(236, 72, 153, 0.5);
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            text-align: center;
            display: none;
            animation: fadeIn 0.4s ease-out forwards;
        }

        .result-card h3 {
            font-size: 1rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .result-card .price {
            font-size: 2.5rem;
            font-weight: 800;
            color: #38bdf8;
            margin-top: 0.3rem;
            text-shadow: 0 0 20px rgba(56, 189, 248, 0.5);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="container">
        <div class="header">
            <h1>House Valuation AI</h1>
            <p>Enter the property specifications to generate a precision valuation estimate</p>
        </div>

        <form id="predictionForm">
            <div class="grid">
                {% for field in fields %}
                <div class="input-group">
                    <label for="{{ field.name }}">{{ field.label }}</label>
                    {% if field.type == 'select' %}
                        <select class="input-control" id="{{ field.name }}" name="{{ field.name }}">
                            {% for opt in field.options %}
                                <option value="{{ opt.val }}">{{ opt.label }}</option>
                            {% endfor %}
                        </select>
                    {% else %}
                        <input class="input-control" 
                               type="number" 
                               id="{{ field.name }}" 
                               name="{{ field.name }}" 
                               value="{{ field.default }}"
                               {% if field.min is defined %}min="{{ field.min }}"{% endif %}
                               {% if field.max is defined %}max="{{ field.max }}"{% endif %}
                               step="{{ field.step }}" required>
                    {% endif %}
                </div>
                {% endfor %}
                <button type="submit" class="btn-submit">Calculate Valuation</button>
            </div>
        </form>

        <div class="result-card" id="resultCard">
            <h3>Estimated Market Value</h3>
            <div class="price" id="predictedPrice">$0</div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((val, key) => data[key] = parseFloat(val));

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const res = await response.json();
            const resultCard = document.getElementById('resultCard');
            const priceEl = document.getElementById('predictedPrice');

            if (res.success) {
                const formattedPrice = new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD',
                    maximumFractionDigits: 0
                }).format(res.prediction);
                
                priceEl.innerText = formattedPrice;
                resultCard.style.display = 'block';
            } else {
                alert('Error generating prediction: ' + res.error);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, fields=FEATURE_CONFIG)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model file (model.pkl) not found on server.'}), 500

    try:
        data = request.get_json()
        
        # Extract features in exact order required by the model
        input_vector = [float(data.get(feat, 0)) for feat in FEATURE_NAMES]
        
        # Reshape for single sample prediction
        features_array = np.array([input_vector])
        
        # Perform prediction
        prediction = model.predict(features_array)[0]
        
        return jsonify({'success': True, 'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
