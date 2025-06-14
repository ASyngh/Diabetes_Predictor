from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('models/diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Feature list (used in both routes)
features = [
    ("Pregnancies", "count"),
    ("Glucose", "mg/dL"),
    ("BloodPressure", "mm Hg"),
    ("SkinThickness", "mm"),
    ("Insulin", "µU/mL"),
    ("BMI", "kg/m²"),
    ("DiabetesPedigreeFunction", "ratio"),
    ("Age", "years")
]

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    input_values = [float(request.form.get(f"f{i}")) for i in range(8)]
    scaled_input = scaler.transform([input_values])
    prediction = model.predict(scaled_input)
    result = "Positive" if prediction[0] == 1 else "Negative"
    return render_template('index.html', prediction_text=f'Diabetes test result: {result}', features=features)

if __name__ == '__main__':
    app.run(debug=True)
