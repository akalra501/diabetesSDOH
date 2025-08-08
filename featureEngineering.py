import pandas as pd
import numpy as np

# --- Step 1: Load the Merged Dataset ---
# This script starts with the big merged file from the first step.
print("Loading the merged NHANES dataset...")
try:
    df = pd.read_csv('nhanesMerge1999_2023.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'nhanesMerge1999_2023.csv' not found.")
    exit()

# --- Step 2: Create the Multi-Class Target Variable ---
print("Target Variable")
conditions = [
    (df['DiagnosedDiabetesRaw'] == 1.0), # Diabetic
    (df['FastingGlucose'] >= 100) & (df['FastingGlucose'] < 130),      # Pre-Diabetic
    (df['FastingGlucose'] < 100)                                       # Non-Diabetic
]
choices = [2, 1, 0] # 2=Diabetic, 1=Pre-Diabetic, 0=Non-Diabetic
df['diabetesCategory'] = np.select(conditions, choices, default=np.nan)

# This makes our final plots and interpretations much easier to understand.
print("Mapping categorical codes to readable names...")
educationMap = {
    1.0: 'LessThan9thGrade',
    2.0: '9-11thGrade',
    3.0: 'HighSchoolGrad',
    4.0: 'SomeCollege',
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

#Creating bins in age groups to see differences in groups.
ageBins = [0, 39, 59, 100]
ageLabels = ['Adult_18-39', 'MiddleAged_40-59', 'Senior_60+']
df['ageGroup'] = pd.cut(df['Age'], bins=ageBins, labels=ageLabels, right=False)

# This prevents a few very high values from skewing the model.
incomeCap = df['IncomePovertyRatio'].quantile(0.95)
df['incomePovertyRatioCapped'] = df['IncomePovertyRatio'].clip(upper=incomeCap)
print(f"Capped IncomePovertyRatio at the 95th percentile: {incomeCap:.2f}")

#Final features to be used for the model.
featuresToUse = [
    'ageGroup',
    'Gender',
    'incomePovertyRatioCapped',
    'educationLevelMapped',
    'raceEthnicityMapped',
    'HasInsurance',
    'diabetesCategory'
]

engineeredDf = df[featuresToUse].copy()
engineeredDf.dropna(inplace=True) # Drop rows if any of our selected features are missing
engineeredDf['diabetesCategory'] = engineeredDf['diabetesCategory'].astype(int)

# This clean, engineered dataset is the input for our modeling script.
outputFilename = 'engineeredData.csv'
engineeredDf.to_csv(outputFilename, index=False)

print(f"\nFeature engineering complete. Cleaned data saved to '{outputFilename}'.")
print("\nFinal data columns:", engineeredDf.columns.tolist())
print("\nFirst 5 rows of engineered data:")
print(engineeredDf.head())

