from flask import Flask, request, render_template_string
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model
MODEL_PATH = "linear_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    model = None
    print("Model Loading Error:", e)

FEATURES = [
    "Number of Bedrooms",
    "Number of Bathrooms",
    "Living Area (sq ft)",
    "Lot Area (sq ft)",
    "Number of Floors",
    "Waterfront Present (0 = No, 1 = Yes)",
    "Number of Views",
    "Condition of the House",
    "Grade of the House",
    "Area of the House (Excluding Basement)",
    "Area of the Basement",
    "Built Year",
    "Renovation Year",
    "Lot Area Renovated",
    "Number of Schools Nearby",
    "Distance from the Airport"
]

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>House Price Prediction</title>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap" rel="stylesheet">

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:'Poppins',sans-serif;
}

body{

background:linear-gradient(135deg,#0f172a,#1e3a8a,#6d28d9);

background-size:300% 300%;
animation:bg 10s infinite alternate;

display:flex;
justify-content:center;
align-items:center;

padding:40px;
min-height:100vh;
}

@keyframes bg{

0%{background-position:left;}
100%{background-position:right;}

}

.container{

width:100%;
max-width:1200px;

background:rgba(255,255,255,0.12);

backdrop-filter:blur(18px);

border-radius:25px;

padding:40px;

box-shadow:
0 20px 50px rgba(0,0,0,.4),
0 0 50px rgba(0,255,255,.15),
0 0 80px rgba(138,43,226,.15);

}

h1{

text-align:center;
color:white;
margin-bottom:30px;
font-size:38px;

}

.grid{

display:grid;
grid-template-columns:repeat(2,1fr);
gap:18px;

}

.card{

display:flex;
flex-direction:column;

}

label{

color:white;
margin-bottom:6px;
font-size:14px;
font-weight:500;

}

input{

padding:13px;
border:none;
outline:none;

border-radius:12px;

background:rgba(255,255,255,.9);

font-size:15px;

transition:.3s;

}

input:focus{

transform:scale(1.03);

box-shadow:0 0 18px cyan;

}

button{

margin-top:30px;

width:100%;

padding:16px;

font-size:20px;

font-weight:bold;

border:none;

border-radius:15px;

cursor:pointer;

color:white;

background:linear-gradient(90deg,#06b6d4,#2563eb,#7c3aed);

box-shadow:
0 0 25px rgba(0,255,255,.4);

transition:.3s;

}

button:hover{

transform:translateY(-3px);

box-shadow:
0 0 40px cyan,
0 0 60px blue;

}

.result{

margin-top:25px;

padding:20px;

border-radius:15px;

background:rgba(255,255,255,.18);

color:white;

font-size:26px;

font-weight:bold;

text-align:center;

box-shadow:0 0 30px rgba(0,255,255,.3);

}

.error{

margin-top:20px;
color:#ffcccc;
font-size:18px;
text-align:center;

}

@media(max-width:850px){

.grid{

grid-template-columns:1fr;

}

}

</style>

</head>

<body>

<div class="container">

<h1>🏡 House Price Prediction System</h1>

<form method="POST">

<div class="grid">

{% for feature in features %}

<div class="card">

<label>{{feature}}</label>

<input type="number" step="any" name="{{feature}}" required>

</div>

{% endfor %}

</div>

<button type="submit">Predict House Price</button>

</form>

{% if prediction %}

<div class="result">

Predicted House Price

<br><br>

₹ {{ prediction }}

</div>

{% endif %}

{% if error %}

<div class="error">

{{error}}

</div>

{% endif %}

</div>

</body>

</html>

"""

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:

            values = []

            for feature in FEATURES:
                values.append(float(request.form[feature]))

            df = pd.DataFrame([values], columns=FEATURES)

            pred = model.predict(df)[0]

            prediction = f"{pred:,.2f}"

        except Exception as e:
            error = str(e)

    return render_template_string(
        HTML,
        features=FEATURES,
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
