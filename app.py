from flask import Flask, request, render_template
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('loan_eligibility_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [float(x) for x in request.form.values()]
        
        # Scale features
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        output = "Eligible" if prediction[0] == 1 else "Not Eligible"
    except Exception as e:
        output = f"Error: {e}"

    return render_template('index.html', prediction_text=f'Loan Prediction: {output}')

if __name__ == '__main__':
    app.run(debug=True)
