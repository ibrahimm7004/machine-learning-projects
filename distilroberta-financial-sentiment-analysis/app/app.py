from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
headers = {"Authorization": "Bearer "}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        prompt = request.form['prompt']
        output = func(prompt)
        data = output[0]  # Extract the first element of the output list
        labels = [item['label'] for item in data]
        percentages = ["{:.5f}".format(item['score']) for item in data]

        return render_template('result.html', labels=labels, percentages=percentages)


def func(prompt):
    output = query({
        "inputs": prompt,
    })
    return output


if __name__ == '__main__':
    app.run(debug=True)
