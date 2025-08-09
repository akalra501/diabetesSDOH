import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === 1. Load senior citizen dataset ===
try:
    df = pd.read_csv('60+/nhanesSeniorCitizens.csv')
except FileNotFoundError:
    print("Error: 'nhanesSeniorCitizenData.csv' not found.")
    exit()

# === 2. Print missingness info ===
print("\nMissing values and counts per column:\n")
for col in df.columns:
    n_missing = df[col].isna().sum()
    n_non_missing = df[col].notna().sum()
    print(f"{col}: Missing = {n_missing}, Non-missing = {n_non_missing}")

# === 3. Output directory for plots ===
outputDir = '60+plots/seniorCitizenVisuals'
os.makedirs(outputDir, exist_ok=True)
print(f"\nPlots will be saved in the '{outputDir}' directory.")

# === 4. Target mapping ===
diabetesLabelMap = {0: 'Non-Diabetic', 1: 'Diabetic'}
if 'isDiabetic' in df.columns:
    df['diabetesStatusMapped'] = df['isDiabetic'].map(diabetesLabelMap)

# === 5. Helper for percentage annotations ===
def addStatsToCategoricalPlot(ax, df, x_var):
    totals = df[x_var].value_counts()
    for p in ax.patches:
        category_index = round(p.get_x() + p.get_width() / 2.)
        x_tick_label = ax.get_xticklabels()[category_index]
        x_category = x_tick_label.get_text()
        height = p.get_height()
        if height > 0 and x_category in totals:
            percentage = 100 * height / totals[x_category]
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 5,
                    f'{percentage:.1f}%',
                    ha="center", va="bottom", fontsize=8)

# === 6. Age group plot (should all be Senior_60+) ===
if 'ageGroup' in df.columns:
    plt.figure(figsize=(6, 5))
    axAge = sns.countplot(data=df, x='ageGroup', hue='diabetesStatusMapped',
                          palette='viridis')
    plt.title('Diabetic vs. Non-Diabetic Status (Senior Citizens)')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    addStatsToCategoricalPlot(axAge, df, 'ageGroup')
    plt.tight_layout()
    plt.savefig(f'{outputDir}/ageGroup_distribution.png')
    plt.show()

# === 7. Education levels ===
if 'educationLevelMapped' in df.columns:
    plt.figure(figsize=(10, 6))
    axEdu = sns.countplot(data=df, x='educationLevelMapped', hue='diabetesStatusMapped',
                          palette='plasma')
    plt.title('Diabetic vs. Non-Diabetic Status Across Education Levels (60+)')
    plt.xlabel('Education Level')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    addStatsToCategoricalPlot(axEdu, df, 'educationLevelMapped')
    plt.tight_layout()
    plt.savefig(f'{outputDir}/educationLevel_distribution.png')
    plt.show()

# === 8. Race / ethnicity ===
if 'raceEthnicityMapped' in df.columns:
    plt.figure(figsize=(10, 6))
    axRace = sns.countplot(data=df, x='raceEthnicityMapped', hue='diabetesStatusMapped',
                           palette='magma')
    plt.title('Diabetic vs. Non-Diabetic Status Across Race/Ethnicity (60+)')
    plt.xlabel('Race / Ethnicity')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    addStatsToCategoricalPlot(axRace, df, 'raceEthnicityMapped')
    plt.tight_layout()
    plt.savefig(f'{outputDir}/raceEthnicity_distribution.png')
    plt.show()

# === 9. Insurance status ===
if 'HasInsurance' in df.columns:
    df['hasInsuranceMapped'] = df['HasInsurance'].map({0: 'No', 1: 'Yes'})
    plt.figure(figsize=(6, 5))
    axIns = sns.countplot(data=df, x='hasInsuranceMapped', hue='diabetesStatusMapped',
                          palette='coolwarm')
    plt.title('Diabetic vs. Non-Diabetic Status by Insurance Coverage (60+)')
    plt.xlabel('Has Insurance')
    plt.ylabel('Count')
    addStatsToCategoricalPlot(axIns, df, 'hasInsuranceMapped')
    plt.tight_layout()
    plt.savefig(f'{outputDir}/insurance_distribution.png')
    plt.show()

# === 10. Income-to-poverty ratio ===
if 'incomePovertyRatioCapped' in df.columns:
    plt.figure(figsize=(10, 6))
    axIncome = sns.histplot(data=df, x='incomePovertyRatioCapped',
                            hue='diabetesStatusMapped',
                            multiple='stack', palette='cividis', bins=20)
    plt.title('Diabetic vs. Non-Diabetic Status by Income-to-Poverty Ratio (60+)')
    plt.xlabel('Income-to-Poverty Ratio (Capped at 95th percentile)')
    plt.ylabel('Count')

    # Stats annotation
    statsText = "Income Ratio Stats (60+):\n\n"
    for status, groupDf in df.groupby('diabetesStatusMapped'):
        meanVal = groupDf['incomePovertyRatioCapped'].mean()
        medianVal = groupDf['incomePovertyRatioCapped'].median()
        stdVal = groupDf['incomePovertyRatioCapped'].std()
        statsText += f"--- {status} ---\n"
        statsText += f"Mean: {meanVal:.2f}\nMedian: {medianVal:.2f}\nStd Dev: {stdVal:.2f}\n\n"

    plt.text(0.98, 0.95, statsText, transform=axIncome.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{outputDir}/incomePovertyRatio_distribution.png')
    plt.show()
