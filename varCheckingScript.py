import pandas as pd
import glob

# Define expected variables for each type of file
expected_columns = {
    "demo": ['RIDAGEYR', 'RIAGENDR', 'INDFMPIR', 'DMDEDUC2', 'RIDRETH1'],
    "diq": ['DIQ010'],
    "hiq": ['HIQ011'],
    "glu": ['LBXGLU']
}

# Define glob patterns for each
file_patterns = {
    "demo": "demoData*.xpt",
    "diq": "diqData*.xpt",
    "hiq": "hiqData*.xpt",
    "glu": "gluData*.xpt"
}

# Loop through all file types
for key in file_patterns:
    print(f"\n--- Checking {key.upper()} files ---")
    for file in sorted(glob.glob(file_patterns[key])):
        df = pd.read_sas(file, format='xport')
        file_vars = df.columns.tolist()
        print(f"\n{file}")
        for var in expected_columns[key]:
            status = "FOUND" if var in file_vars else "MISSING"
            print(f"  {var}: {status}")
