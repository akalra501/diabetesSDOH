import pandas as pd
import numpy as np

# Load the raw, merged NHANES dataset.
print("Loading the raw NHANES dataset...")
try:
    # This assumes your raw data is in this file from the very first script.
    df = pd.read_csv('nhanesMerge1999_2023.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'nhanesMerge1999_2023.csv' not found. Please ensure your raw merged data file is present.")
    exit()

# Engineer the BINARY 'isDiabetic' target variable.
# We will only use explicitly diagnosed cases for a clear separation.
# 1 = Diabetic, 0 = Non-Diabetic. We will ignore borderline cases for this analysis.
print("\nEngineering the binary 'isDiabetic' target variable...")
df['isDiabetic'] = df['DiagnosedDiabetesRaw'].map({1.0: 1, 2.0: 0})

# Map numerical codes to readable string labels for better interpretation.
print("Mapping categorical codes to readable names...")
educationMap = {
    1.0: 'LessThan9thGrade',
    2.0: '9-11thGrade',
    3.0: 'HighSchoolGrad_GED',
    4.0: 'SomeCollege_AA',
    5.0: 'CollegeGradOrAbove'
}
raceMap = {
    1.0: 'MexicanAmerican',
    2.0: 'OtherHispanic',
    3.0: 'NonHispanicWhite',
    4.0: 'NonHispanicBlack',
    5.0: 'OtherRace'
}
df['educationLevelMapped'] = df['EducationLevel'].map(educationMap)
df['raceEthnicityMapped'] = df['RaceEthnicity'].map(raceMap)

# Create new features from the existing data to improve model performance.
print("Performing feature engineering...")
ageBins = [0, 39, 59, 100]
ageLabels = ['Adult_18-39', 'MiddleAged_40-59', 'Senior_60+']
df['ageGroup'] = pd.cut(df['Age'], bins=ageBins, labels=ageLabels, right=False)

incomeCap = df['IncomePovertyRatio'].quantile(0.95)
df['incomePovertyRatioCapped'] = df['IncomePovertyRatio'].clip(upper=incomeCap)

# Select the final features for the model.
featuresToUse = [
    'ageGroup',
    'Gender',
    'incomePovertyRatioCapped',
    'educationLevelMapped',
    'raceEthnicityMapped',
    'HasInsurance',
    'isDiabetic' # Using our new binary target
]

engineeredDf = df[featuresToUse].copy()

# Drop any rows where our target or key features are missing.
engineeredDf.dropna(inplace=True)
engineeredDf['isDiabetic'] = engineeredDf['isDiabetic'].astype(int)

# Save the final engineered data to a new CSV file.
outputFilename = 'engineeredBinaryFeatures.csv'
engineeredDf.to_csv(outputFilename, index=False)

print(f"\nFeature engineering for binary classification complete. Cleaned data saved to '{outputFilename}'.")
