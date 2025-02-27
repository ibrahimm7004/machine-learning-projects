from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
# Load the model, and the encoder and scaler objects
model = joblib.load(r'C:\Users\hp\Desktop\codeNinja\week10\app1\model_files\model.joblib')  
encoder = joblib.load(r'C:\Users\hp\Desktop\codeNinja\week10\app1\model_files\encoder.joblib')  
scaler = joblib.load(r'C:\Users\hp\Desktop\codeNinja\week10\app1\model_files\scaler.joblib')  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    column_headers = ["number_of_orders", "no_of_products", "total_sales", "gmv", ]

    df = pd.DataFrame(columns=column_headers)

    form_data = request.form
    # Convert form data into a dictionary
    data_dict = dict(form_data)
    
    # Append the form data as a new row to the DataFrame
    df = df.append(data_dict, ignore_index=True)

    return render_template('index.html', prediction_text=df)

if __name__ == "__main__":
    app.run(debug=True)