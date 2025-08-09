import pandas as pd
import glob
import re

#loading in all the files.
demo_files = sorted(glob.glob("data/demoData*.xpt"))
diq_files  = sorted(glob.glob("data/diqData*.xpt"))
hiq_files  = sorted(glob.glob("data/hiqData*.xpt"))
glu_files  = sorted(glob.glob("data/gluData*.xpt"))

#loading helper function, and keeping certain columns.
def load_and_trim(file, cycle, columns):
    """Load file, keep only selected columns, and tag with cycle."""
    df = pd.read_sas(file, format='xport')
    df = df[columns]
    df['cycle'] = cycle
    return df

#creating empy lists for easy iteration.
demo_data = []
diq_data  = []
hiq_data  = []
glu_data  = []

#adding relevant columns to our lists.
for file in demo_files:
    cycle = re.search(r"demoData(\d{4}_\d{4})", file).group(1)
    df = load_and_trim(file, cycle, [
        'SEQN', 'RIDAGEYR', 'RIAGENDR', 'INDFMPIR', 'DMDEDUC2', 'RIDRETH1'
    ])
    demo_data.append(df)


for file in diq_files:
    cycle = re.search(r"diqData(\d{4}_\d{4})", file).group(1)
    df = load_and_trim(file, cycle, ['SEQN', 'DIQ010'])
    diq_data.append(df)

for file in glu_files:
    cycle = re.search(r"gluData(\d{4}_\d{4})", file).group(1)
    df = load_and_trim(file, cycle, ['SEQN', 'LBXGLU'])
    glu_data.append(df)

for file in hiq_files:
    cycle = re.search(r"hiqData(\d{4}_\d{4})", file).group(1)
    df = pd.read_sas(file, format='xport')
    if 'HID010' in df.columns:
        df = df[['SEQN', 'HID010']].rename(columns={'HID010': 'HasInsuranceRaw'})
    elif 'HIQ011' in df.columns:
        df = df[['SEQN', 'HIQ011']].rename(columns={'HIQ011': 'HasInsuranceRaw'})
    else:
        print(f"Skipping {file} — no insurance variable found.")
        continue
    df['cycle'] = cycle
    hiq_data.append(df)
    
print("Demo files found:", demo_files)
print("DIQ files found:", diq_files)
print("HIQ files found:", hiq_files)
print("GLU files found:", glu_files)

print("Demo dataframes loaded:", len(demo_data))
print("DIQ dataframes loaded:", len(diq_data))
print("HIQ dataframes loaded:", len(hiq_data))
print("GLU dataframes loaded:", len(glu_data))
#combining data from all the lists.
demo_df = pd.concat(demo_data, ignore_index=True)
diq_df  = pd.concat(diq_data, ignore_index=True)
hiq_df  = pd.concat(hiq_data, ignore_index=True)
glu_df  = pd.concat(glu_data, ignore_index=True)


#merging into a singular dataset.
merged = demo_df.merge(diq_df, on=['SEQN', 'cycle'], how='inner')
merged = merged.merge(hiq_df, on=['SEQN', 'cycle'], how='left')  
merged = merged.merge(glu_df, on=['SEQN', 'cycle'], how='left')

#cleaning and mappting variables.
merged['diabetes'] = merged['DIQ010'].map({1: 1, 2: 0})
merged['Gender'] = merged['RIAGENDR'].map({1: 'Male', 2: 'Female'})

#unifying insurance values (1 = Yes, 0 = No)
merged['HasInsurance'] = merged['HasInsuranceRaw'].map({1: 1, 2: 0})

#renaming columns for clarity
merged.rename(columns={
    'RIDAGEYR': 'Age',
    'RIAGENDR': 'GenderCode',
    'INDFMPIR': 'IncomePovertyRatio',
    'DMDEDUC2': 'EducationLevel',
    'RIDRETH1': 'RaceEthnicity',
    'LBXGLU': 'FastingGlucose',
    'DIQ010': 'DiagnosedDiabetesRaw',
    'diabetes': 'Diabetes',
    'cycle': 'SurveyCycle'
}, inplace=True)

#merging into one file.
merged.to_csv("nhanesMerge1999_2023_REVISED.csv", index=False)
