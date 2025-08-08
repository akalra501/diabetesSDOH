import pandas as pd
import glob

hiq_files = sorted(glob.glob("hiqData*.xpt"))

for file in hiq_files:
    print(f"\n--- {file} ---")
    try:
        df = pd.read_sas(file, format='xport')
        print("Variables:", list(df.columns))
        print("\nFirst 5 rows:")
        print(df.head())  # Show first 5 rows
    except Exception as e:
        print(f"Error reading {file}: {e}")
