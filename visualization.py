import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the engineered dataset with the 3-class diabetic target.
try:
    # This assumes the 3-class data was saved to 'engineered_features.csv'
    df = pd.read_csv('engineeredData.csv')
except FileNotFoundError:
    print("Error: 'engineeredData.csv' not found.")
    exit()

# Create a dedicated directory for these new plots.
outputDir = 'plots/engineered3ClassVisuals'
os.makedirs(outputDir, exist_ok=True)
print(f"Plots saved in the '{outputDir}' directory.")

# Map the numeric target back to readable labels for clear plot legends.
diabetesLabelMap = {
    0: 'Non-Diabetic',
    1: 'Pre-Diabetic',
    2: 'Diabetic'
}
df['diabetesStatusMapped'] = df['diabetesCategory'].map(diabetesLabelMap)

# Helper function to add percentage annotations to the bars.
def addStatsToCategoricalPlot(ax, df, x_var):
    """Adds percentage annotations to each bar in a countplot."""
    totals = df[x_var].value_counts()
    for p in ax.patches:
        # Using round() for more accurate index lookup.
        category_index = round(p.get_x() + p.get_width() / 2.)
        x_tick_label = ax.get_xticklabels()[category_index]
        x_category = x_tick_label.get_text()
        
        height = p.get_height()
        if height > 0:
            percentage = 100 * height / totals[x_category]
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 5,
                    f'{percentage:.1f}%',
                    ha="center", va="bottom", fontsize=8)

# Create a count plot for the Age Groups.
plt.figure(figsize=(10, 7))
axAge = sns.countplot(data=df, x='ageGroup', hue='diabetesStatusMapped', order=['Adult_18-39', 'MiddleAged_40-59', 'Senior_60+'], palette='viridis')
plt.title('Distribution of Diabetes Status Across Age Groups (3-Class)')
plt.xlabel('Age Group')
plt.ylabel('Count')
addStatsToCategoricalPlot(axAge, df, 'ageGroup')
plt.tight_layout()
plt.savefig(f'{outputDir}/ageGroup_3_class_distribution.png')
plt.show()

# Create a count plot for the Education Levels.
plt.figure(figsize=(12, 8))
axEdu = sns.countplot(data=df, x='educationLevelMapped', hue='diabetesStatusMapped', palette='plasma',
                       order=['LessThan9thGrade', '9-11thGrade', 'HighSchoolGrad', 'SomeCollege', 'CollegeGradOrAbove'])
plt.title('Distribution of Diabetes Status Across Education Levels (3-Class)')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
addStatsToCategoricalPlot(axEdu, df, 'educationLevelMapped')
plt.tight_layout()
plt.savefig(f'{outputDir}/educationLevel_3_class_distribution.png')
plt.show()

# Create a count plot for Race and Ethnicity.
plt.figure(figsize=(12, 8))
axRace = sns.countplot(data=df, x='raceEthnicityMapped', hue='diabetesStatusMapped', palette='magma',
                        order=['NonHispanicWhite', 'NonHispanicBlack', 'MexicanAmerican', 'OtherHispanic', 'OtherRace'])
plt.title('Distribution of Diabetes Status Across Race/Ethnicity (3-Class)')
plt.xlabel('Race / Ethnicity')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
addStatsToCategoricalPlot(axRace, df, 'raceEthnicityMapped')
plt.tight_layout()
plt.savefig(f'{outputDir}/raceEthnicity_3_class_distribution.png')
plt.show()

# Create a count plot for Insurance Status.
plt.figure(figsize=(8, 6))
df['hasInsuranceMapped'] = df['HasInsurance'].map({0.0: 'No', 1.0: 'Yes'})
axIns = sns.countplot(data=df, x='hasInsuranceMapped', hue='diabetesStatusMapped', palette='coolwarm', order=['No', 'Yes'])
plt.title('Distribution of Diabetes Status by Insurance Coverage (3-Class)')
plt.xlabel('Has Insurance')
plt.ylabel('Count')
addStatsToCategoricalPlot(axIns, df, 'hasInsuranceMapped')
plt.tight_layout()
plt.savefig(f'{outputDir}/insurance_3_class_distribution.png')
plt.show()

# Create a histogram for the capped Income-to-Poverty Ratio.
plt.figure(figsize=(12, 7))
axIncome = sns.histplot(data=df, x='incomePovertyRatioCapped', hue='diabetesStatusMapped', multiple='stack', palette='cividis', bins=20)
plt.title('Distribution of Diabetes Status by Income-to-Poverty Ratio (3-Class)')
plt.xlabel('Income-to-Poverty Ratio (Capped at 95th percentile)')
plt.ylabel('Count')

# Add a text box with stats for the continuous variable.
statsText = "Income Ratio Stats:\n\n"
for status, groupDf in df.groupby('diabetesStatusMapped'):
    meanVal = groupDf['incomePovertyRatioCapped'].mean()
    medianVal = groupDf['incomePovertyRatioCapped'].median()
    stdVal = groupDf['incomePovertyRatioCapped'].std()
    statsText += f"--- {status} ---\n"
    statsText += f"Mean: {meanVal:.2f}\nMedian: {medianVal:.2f}\nStd Dev: {stdVal:.2f}\n\n"

plt.text(0.7, 0.95, statsText, transform=axIncome.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{outputDir}/incomePovertyRatio_3_class_distribution.png')
plt.show()

