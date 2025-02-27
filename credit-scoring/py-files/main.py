

# Function to train a Linear Regression model
def train_linear_regression(X_train, y_train, X_test, y_test):
    # Create a Linear Regression model
    model = LinearRegression()
    # Train the model using training data
    model.fit(X_train, y_train)
    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Return true labels, predicted labels, and model name
    return model, y_test, y_pred, "Linear Regression"

# Function to train a Random Forest model


def train_random_forest(X_train, y_train, X_test, y_test):
    # Create a Random Forest model
    model = RandomForestRegressor()
    # Train the model using training data
    model.fit(X_train, y_train)
    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Return true labels, predicted labels, and model name
    return model, y_test, y_pred, "Random Forest"

# Function to train an XGBoost model


def train_xgboost(X_train, y_train, X_test, y_test):
    # Create an XGBoost model
    model = XGBRegressor()
    # Train the model using training data
    model.fit(X_train, y_train)
    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Return true labels, predicted labels, and model name
    return model, y_test, y_pred, "XGBoost"

# Function to train a Neural Network model


def train_neural_network(X_train, y_train, X_test, y_test):
    # Create a Neural Network model
    model = MLPRegressor()
    # Train the model using training data
    model.fit(X_train, y_train)
    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Return true labels, predicted labels, and model name
    return model, y_test, y_pred, "Neural Network"

# Function to train a Support Vector Machine (SVM) model


def train_svm(X_train, y_train, X_test, y_test):
    # Create an SVM model
    model = SVR()
    # Train the model using training data
    model.fit(X_train, y_train)
    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Return true labels, predicted labels, and model name
    return model, y_test, y_pred, "Support Vector Machine"

# Function to train a K-Nearest Neighbors (KNN) model


def train_knn(X_train, y_train, X_test, y_test):
    # Create a KNN model
    model = KNeighborsRegressor()
    # Train the model using training data
    model.fit(X_train, y_train)
    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Return true labels, predicted labels, and model name
    return model, y_test, y_pred, "K-Nearest Neighbours"


def evaluate_metrics(y_test, y_pred, model_name):
    model_metrics = []
    n_features = 12     # No. of features being used

    # Calculate the Mean Squared Error (MSE) between true labels (y_test) and predicted labels (y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Calculate the Root Mean Squared Error (RMSE) by taking the square root of the MSE
    rmse = np.sqrt(mse)

    # Calculate the Mean Absolute Error (MAE) between true labels (y_test) and predicted labels (y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate the R-squared (coefficient of determination) score between true labels (y_test) and predicted labels (y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate the Adjusted R-squared score, which adjusts R-squared based on the number of features used in the model
    # It penalizes overfitting and takes into account the number of samples (len(y_test)) and number of features (n_features)
    adjusted_r2 = 1 - ((1 - r2) * (len(y_test) - 1) /
                       (len(y_test) - n_features - 1))

    # Append all metrics to the model_metrics list
    model_metrics.append(model_name)
    model_metrics.append(mse)
    model_metrics.append(rmse)
    model_metrics.append(mae)
    model_metrics.append(r2)
    model_metrics.append(adjusted_r2)

    return model_metrics


def create_folder(directory, name):
    folder_path = os.path.join(directory, name)

    if os.path.exists(folder_path):
        try:
            # Delete the existing folder and its contents
            shutil.rmtree(folder_path)
        except OSError as e:
            print(f"Error: {e}")

    try:
        os.mkdir(folder_path)  # Create the new folder
    except OSError as e:
        print(f"Error: {e}")

    return folder_path


def create_test_data_csv(df, directory):
    folder_path = create_folder(directory, 'excel_files')
    new_df = remove_rows_with_nan(df)
    X_train, X_test, y_train, y_test = perform_data_split(new_df)

    # Create DataFrames for test data
    x_columns = pd.DataFrame(X_test)
    y_columns = pd.DataFrame(y_test)

    # Combine the x and y dataframes
    test_data_df = pd.concat([x_columns, y_columns], axis=1)

    # Save the test data DataFrame to a new CSV file
    test_data_df.to_csv(os.path.join(
        folder_path, 'test_data.csv'), index=False)


def execute_models(df, directory):
    functions_to_execute = [train_linear_regression, train_random_forest,
                            train_xgboost, train_neural_network, train_svm, train_knn]
    metrics = []
    max = 0
    # Loop through each model function and execute it
    for func in functions_to_execute:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = perform_data_split(df)

        # Train the model, make predictions, and get model name
        model, y_true, y_pred, name = func(X_train, y_train, X_test, y_test)

        # Calculate evaluation metrics for the model's predictions
        model_metrics = evaluate_metrics(y_true, y_pred, name)

        # Check which model is most accurate using r2-squared value
        if model_metrics[4:5][0] > max:
            max = model_metrics[4:5][0]
            # Store the most efficient model and its name
            efficient_model, efficient_model_name = model, name

        # Append the model's metrics (excluding the name) to the list
        metrics.append([name] + model_metrics[1:])

    create_df(metrics)
    return efficient_model, efficient_model_name


def create_df(model_metrics):
    # Define column names for the DataFrame
    columns = ['Model', 'MSE', 'RMSE', 'MAE', 'R-2', 'Adjusted R-2']

    # Create a new DataFrame using the metrics data and column names
    new_df = pd.DataFrame(model_metrics, columns=columns)

    print("\n\n\n", new_df, "\n\n")


def main():
    # Set your directory:
    existing_directory = r'C:\Users\hp\Desktop\codeNinja\week8and9\credit_scoring'

    df, numerical_features, categorical_features = initial_tasks(
        existing_directory)

    # Preprocessing steps:
    processed_df = preprocess(df)
    encoded_df, encoder = encoding(processed_df, categorical_features)
    preprocessed_df = remove_rows_with_nan(encoded_df)
    normalized_df, scaler = normalization(preprocessed_df, numerical_features)

    # Storing fully pre-processed dataframe to a new CSV:
    # normalized_df.to_csv(os.path.join(existing_directory, 'output.csv'), index=False)

    # Splitting data, training models, making predictions, and evaluating model accuracy:
    model, name = execute_models(normalized_df, existing_directory)

    # Create a new folder named 'model_files'
    folder_path = create_folder(existing_directory, 'model_files')

    # Store the model and preprocessing steps to joblib files in the newly created folder
    store_model(model, name, folder_path)
    encoded_df, encoder = encoding(processed_df, categorical_features)
    store_preprocessors(encoder, scaler, folder_path)


main()
