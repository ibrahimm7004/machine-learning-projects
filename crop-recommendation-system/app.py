from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the SVM pipeline
svm_pipeline = joblib.load(
    r'C:\Users\hp\Desktop\sem6\PL Lab\mid\svm_pipeline.joblib')
scaler = svm_pipeline.named_steps['scaler']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        user_input = {
            'N': float(request.form['N']),
            'P': float(request.form['P']),
            'K': float(request.form['K']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall'])
        }

        # Convert user input to a DataFrame
        user_df = pd.DataFrame([user_input])

        try:
            # Check if the scaler has been fitted, if not, fit it
            scaler.transform(user_df)
        except AttributeError:
            scaler.fit(user_df)

        # Perform prediction using the loaded SVM pipeline
        prediction = svm_pipeline.predict(user_df)[0]

        return render_template('index.html', prediction=f'The predicted crop is: {prediction}')


if __name__ == '__main__':
    app.run(debug=True)
