import pandas as pd
import os

BASE_DIR = r"e:\PROJECT\INFOSYS-AI"
data_path = os.path.join(BASE_DIR, "backend", "data", "player_transfer_value_with_sentiment.csv")

print(f"Checking {data_path}...")
if os.path.exists(data_path):
    try:
        df = pd.read_csv(data_path)
        print(f"Success: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()[:5]}...")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("File not found")
