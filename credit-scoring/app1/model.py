import pandas as pd
import re

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


def apply_model(df, model):
    # Make predictions
    predictions = model.predict(df)

    return predictions


def initial_tasks():
    # List of numerical features for scaling
    numerical_features = ['number_of_orders', 'no_of_products', 'total_sales', 'gmv',
        'avg_daily_sales', 'aprox_exist_inventory', 'shop_size',
        'business_age(year)', 'electricity_bill', 'rent_amount']
    
    # Get the categorical columns
    categorical_features = ['shop_type', 'is_rental']

    return numerical_features, categorical_features


def main(df, model, encoder, scaler):
    # Some initial tasks
    numerical_features, categorical_features = initial_tasks()

    # Preprocess the data
    processed_df = preprocess(df)

    # Applying the encoding and scaling, using the loaded objects, on numeric and categorical features accordingly
    processed_data = apply_pipeline_steps(processed_df, encoder, scaler, numerical_features, categorical_features)

    # Load the trained model and apply it to our processed dataset, a new csv named 'predicted_data' will be created 
    apply_model(processed_data, model)

main()
