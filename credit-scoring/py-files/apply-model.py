import joblib
import pandas as pd
import re
import os
import warnings
import sqlite3

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Generalised function to handle and process all differnt data in the columns:


def process_data(column):
    processed_values = []

    for value in column:
        if pd.isna(value):  # Handle empty or NaN values
            processed_values.append(None)
            continue

        # Extract numeric values using regular expression
        numbers = re.findall(r'\d+', value)

        # Handle cases where multiple numeric values are found
        if len(numbers) >= 2:
            start, end = map(int, numbers)
            if 'k' in value.lower():
                start *= 1000
                end *= 1000
            avg = (start + end) / 2
            processed_values.append(int(avg))
        # Handle cases where only one numeric value is found
        elif len(numbers) == 1:
            num = int(numbers[0])
            if 'k' in value.lower():
                num *= 1000
            processed_values.append(num)
        else:
            # Handle cases where numeric values are not directly present
            if 'to' in value.lower() or '-' in value:
                numbers = re.findall(r'\d+', value)
                start, end = map(int, numbers)
                if 'k' in value.lower():
                    start *= 1000
                    end *= 1000
                avg = (start + end) / 2
                processed_values.append(int(avg))
            elif '+' in value:
                num = int(value.replace('+', ''))
                if 'k' in value.lower():
                    num *= 1000
                processed_values.append(num)
            elif 'more' in value.lower():
                num = int(numbers[0])
                processed_values.append(num)
            elif 'less' in value.lower():
                num = int(numbers[0])
                processed_values.append(num)
            else:
                processed_values.append(None)

    return processed_values

# Function to drop a specific column


def drop_column(data_frame, column_name):
    new_df = data_frame.drop(columns=[column_name])
    return new_df


def preprocess(df):
    # Convert 'total_sales' column to numerical
    df['total_sales'] = df['total_sales'].str.replace(',', '').str.strip()
    df['total_sales'] = pd.to_numeric(df['total_sales'], errors='coerce')

    # Apply the process_data function to multiple columns
    columns_to_process = ['aprox_exist_inventory', 'no_of_products', 'number_of_orders',
                          'avg_daily_sales', 'rent_amount', 'gmv', 'using_pos', 'shop_size',
                          'business_age(year)', 'electricity_bill']

    for column_name in columns_to_process:
        df[column_name] = process_data(df[column_name])

    # Call the function to drop the 'using_pos' and 'credit_score' columns
    new_df = drop_column(df, 'using_pos')
    new_df = drop_column(df, 'credit_score')

    # Convert new_df to DataFrame
    new_df = pd.DataFrame(new_df)

    return new_df  # Return the new DataFrame


def initial_tasks(directory, dataset_to_test):
    # Read the sqlite file
    conn = sqlite3.connect(os.path.join(directory, dataset_to_test))

    # Fetch all data from the table Credit_Scoring
    query = "SELECT * FROM 'Credit_Scoring'"
    # Read the SQL Query, which will get the db as a df
    df = pd.read_sql_query(query, conn)

    # Drop the first column (extra index column)
    df = df.drop(df.columns[0], axis=1)

    conn.close()

    # List of numerical features for scaling
    numerical_features = ['number_of_orders', 'no_of_products', 'total_sales', 'gmv',
                          'avg_daily_sales', 'aprox_exist_inventory', 'shop_size',
                          'business_age(year)', 'electricity_bill', 'rent_amount']

    # Get the categorical columns
    categorical_features = ['shop_type', 'is_rental']

    return df, numerical_features, categorical_features


def load_preprocessors(directory):
    # Load the encoder and scaler objects
    encoder = joblib.load(os.path.join(
        directory, 'model_files/encoder.joblib'))
    scaler = joblib.load(os.path.join(directory, 'model_files/scaler.joblib'))

    return encoder, scaler


def apply_pipeline_steps(df, encoder, scaler, numerical_features, categorical_features):
    # Apply encoding to categorical features
    encoded_data = encoder.transform(df[categorical_features])

    # Get the one-hot encoded feature names
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)

    # Convert encoded_data to a DataFrame with the appropriate column names
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names)

    # Apply scaling to numeric features
    scaled_data = scaler.transform(df[numerical_features])

    # Convert scaled_data to a DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=numerical_features)

    # Combine the encoded and scaled DataFrames
    processed_data = pd.concat([scaled_df, encoded_df], axis=1)

    return processed_data


def read_model_name_from_file(directory, filename):
    file_path = os.path.join(directory, filename)

    with open(file_path, 'r') as file:
        content = file.read()

    return content


def apply_model(df, directory, original_dataset):
    new_directory = os.path.join(directory, 'model_files')
    # Read model name from saved txt
    name = read_model_name_from_file(new_directory, 'model_name.txt')

    # Load the saved pipeline and model
    loaded_model = joblib.load(os.path.join(
        new_directory, '{}.joblib'.format(name)))

    # Make predictions
    predictions = loaded_model.predict(df)

    # Add predictions as a new column to the existing DataFrame 'df'
    original_dataset['credit_score'] = predictions

    # Save the modified DataFrame to a new CSV file
    original_dataset.to_csv(os.path.join(
        directory, 'excel_files/predicted_data.csv'), index=False)


def print_comparison(directory):
    # Read the CSV files
    df1 = pd.read_csv(os.path.join(directory, 'test_data.csv'))
    df2 = pd.read_csv(os.path.join(directory, 'predicted_data.csv'))

    # Print the first 10 rows of the specified columns
    df1 = df1.head(10)[['credit_score']].rename(
        columns={'credit_score': 'Original'})
    df2 = df2.head(10)[['credit_score']].rename(
        columns={'credit_score': 'Predicted'})

    # Combine both dfs
    df = pd.concat([df1, df2], axis=1)

    print("\nComparison of first 10 Rows:\n\n", df)


def main():
    directory = r'credit_scoring'     # Modify directory here
    # Modify dataset name here, if required (csv)
    dataset_name = 'test_data.sqlite'

    # Performing some necessary initial tasks
    df, numerical_features, categorical_features = initial_tasks(
        directory, dataset_name)   # new_data is a df containing the loaded csv data

    store_df = df   # Make another copy of the original dataset, to be used later

    # Preprocessing the data:
    processed_df = preprocess(df)

    # Loading pipeline joblib file to get encoder and scaler objects
    encoder, scaler = load_preprocessors(directory)

    # Applying the encoding and scaling, using the loaded objects, on numeric and categorical features accordingly
    processed_data = apply_pipeline_steps(
        processed_df, encoder, scaler, numerical_features, categorical_features)

    # Storing fully pre-processed dataframe to a new CSV:
    processed_data.to_csv(os.path.join(
        directory, 'excel_files/preprocessed_data.csv'), index=False)

    # Load the trained model and apply it to our processed dataset, a new csv named 'predicted_data' will be created
    apply_model(processed_data, directory, store_df)

    # Print the first 10 credit scores from both the test data and predicted data
    print_comparison(os.path.join(directory, 'excel_files'))


main()
