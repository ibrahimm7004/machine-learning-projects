import sqlite3
import pandas as pd
import os

# Edit directory here (should contain the CreditScoring csv)
directory = r'credit_scoring'
csv_file_name = r'excel_files\test_data.csv'                             # CSV name
# Database will be stored in same folder
database_file_name = 'test_data.sqlite'

df = pd.read_csv(os.path.join(directory, csv_file_name))

conn = sqlite3.connect(os.path.join(directory, database_file_name))

df.to_sql('Credit_Scoring', conn, if_exists='replace')

conn.close()
