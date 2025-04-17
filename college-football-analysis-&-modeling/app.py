from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load preprocessing objects and model
team_name_encoder = joblib.load(
    r'team_name_encoder.joblib')
regression_model = joblib.load(
    r'linear_regression_model.joblib')

# Prediction Function


def predict_probabilities(home_team, away_team):
    teams_to_predict = pd.DataFrame([[home_team, away_team]], columns=[
                                    "_home_team", "_away_team"])

    # One-hot encode team names
    encoded_teams = team_name_encoder.transform(teams_to_predict)

    # Combine encoded teams with other features
    input_data = pd.concat([teams_to_predict.drop(
        ["_home_team", "_away_team"], axis=1), pd.DataFrame(encoded_teams)], axis=1)

    # Make predictions
    predictions = regression_model.predict(input_data)

    return {"Home Team Win Probability": predictions[0, 0], "Away Team Win Probability": predictions[0, 1]}

# Flask route for prediction


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    home_team = request.form['home_team']
    away_team = request.form['away_team']
    predictions = predict_probabilities(home_team, away_team)
    return render_template('index.html', predictions=predictions, home_team=home_team, away_team=away_team)


if __name__ == '__main__':
    app.run(debug=True)
