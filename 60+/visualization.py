import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

#Load dataset
DATA_PATH = '60+/nhanesSeniorCitizens.csv'
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: '{DATA_PATH}' not found.")
    raise SystemExit(1)

#Print missingness info 
print("\nMissing values and counts per column:\n")
for col in df.columns:
    n_missing = df[col].isna().sum()
    n_non_missing = df[col].notna().sum()
    print(f"{col}: Missing = {n_missing}, Non-missing = {n_non_missing}")

#Output directory for plots 
outputDir = '60+plots/seniorCitizenVisuals'
os.makedirs(outputDir, exist_ok=True)
print(f"\nPlots will be saved in the '{outputDir}' directory.")

#4) Target mapping 
diabetesLabelMap = {0: 'Non-Diabetic', 1: 'Diabetic'}
if 'isDiabetic' in df.columns:
    df['diabetesStatusMapped'] = df['isDiabetic'].map(diabetesLabelMap)

#5) Helpers 
def add_within_category_percent_labels(ax, data, x_var):
    """
    Labels each bar with % within its x category (e.g., among 'HighSchool', what % are Diabetic vs Non-Diabetic).
    Assumes a seaborn countplot with hue (dodged bars).
    """
    totals = data[x_var].value_counts()
    for p in ax.patches:
        try:
            cat_index = round(p.get_x() + p.get_width() / 2.)
            x_tick_label = ax.get_xticklabels()[cat_index]
            x_category = x_tick_label.get_text()
        except Exception:
            continue
        height = p.get_height()
        if height > 0 and x_category in totals and totals[x_category] > 0:
            pct = 100.0 * height / totals[x_category]
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 0.01 * max(1.0, ax.get_ylim()[1]),
                    f'{pct:.1f}%',
                    ha="center", va="bottom", fontsize=8)

def add_category_share_of_total_labels(ax, data, x_var):
    """
    Adds one label per x category showing what % of the entire dataset that category constitutes.
    Places the label centered above the tallest bar in that category.
    """
    total_n = len(data)
    if total_n <= 0:
        return

    # count per category
    cat_counts = data[x_var].value_counts()

    # Prepare: find max height for each category to place label above bars
    max_heights = {cat: 0 for cat in cat_counts.index}
    # Map bar heights to categories using tick labels
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    # Build a quick map from cat -> max height observed among its bars
    for p in ax.patches:
        try:
            cat_index = round(p.get_x() + p.get_width() / 2.)
            if 0 <= cat_index < len(tick_labels):
                cat = tick_labels[cat_index]
                max_heights[cat] = max(max_heights.get(cat, 0), p.get_height())
        except Exception:
            continue

    # Now place one label per category
    xticks = ax.get_xticks()
    new_ylim_top = ax.get_ylim()[1]
    for i, cat in enumerate(tick_labels):
        count = cat_counts.get(cat, 0)
        pct_total = 100.0 * count / total_n if total_n > 0 else 0.0
        # y-position just above the tallest bar in this category
        y_pos = max_heights.get(cat, 0) + 0.04 * max(1.0, ax.get_ylim()[1])
        ax.text(xticks[i], y_pos, f'{pct_total:.1f}% of total',
                ha='center', va='bottom', fontsize=8, color='dimgray', fontstyle='italic')
        new_ylim_top = max(new_ylim_top, y_pos * 1.05)

    ax.set_ylim(ax.get_ylim()[0], new_ylim_top)

#6) Age group plot (all should be Senior_60+, but we keep it anyway) 
if 'ageGroup' in df.columns and 'diabetesStatusMapped' in df.columns:
    plt.figure(figsize=(6, 5))
    axAge = sns.countplot(data=df, x='ageGroup', hue='diabetesStatusMapped', palette='viridis')
    plt.title('Diabetic vs. Non-Diabetic Status (Senior Citizens)')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    # Keep original labels: % within each age category
    add_within_category_percent_labels(axAge, df, 'ageGroup')
    # Also: show what share of the whole sample each age group is
    add_category_share_of_total_labels(axAge, df, 'ageGroup')
    plt.tight_layout()
    plt.savefig(f'{outputDir}/ageGroup_distribution.png', dpi=300)
    plt.show()

# === 7) Education levels ===
if 'educationLevelMapped' in df.columns and 'diabetesStatusMapped' in df.columns:
    plt.figure(figsize=(10, 6))
    axEdu = sns.countplot(data=df, x='educationLevelMapped', hue='diabetesStatusMapped', palette='plasma')
    plt.title('Diabetic vs. Non-Diabetic Across Education Levels (60+)')
    plt.xlabel('Education Level')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    # Keep original labels: % within each education category
    add_within_category_percent_labels(axEdu, df, 'educationLevelMapped')
    # Also: share of the whole dataset per education category
    add_category_share_of_total_labels(axEdu, df, 'educationLevelMapped')
    plt.tight_layout()
    plt.savefig(f'{outputDir}/educationLevel_distribution.png', dpi=300)
    plt.show()

# === 8) Race / ethnicity ===
if 'raceEthnicityMapped' in df.columns and 'diabetesStatusMapped' in df.columns:
    plt.figure(figsize=(10, 6))
    axRace = sns.countplot(data=df, x='raceEthnicityMapped', hue='diabetesStatusMapped', palette='magma')
    plt.title('Diabetic vs. Non-Diabetic Across Race/Ethnicity (60+)')
    plt.xlabel('Race / Ethnicity')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    # Keep original labels: % within each race/ethnicity category
    add_within_category_percent_labels(axRace, df, 'raceEthnicityMapped')
    # Also: share of the whole dataset per race/ethnicity
    add_category_share_of_total_labels(axRace, df, 'raceEthnicityMapped')
    plt.tight_layout()
    plt.savefig(f'{outputDir}/raceEthnicity_distribution.png', dpi=300)
    plt.show()

#9) Insurance status 
if 'HasInsurance' in df.columns and 'diabetesStatusMapped' in df.columns:
    df['hasInsuranceMapped'] = df['HasInsurance'].map({0: 'No', 1: 'Yes'})
    plt.figure(figsize=(6, 5))
    axIns = sns.countplot(data=df, x='hasInsuranceMapped', hue='diabetesStatusMapped', palette='coolwarm')
    plt.title('Diabetic vs. Non-Diabetic by Insurance Coverage (60+)')
    plt.xlabel('Has Insurance')
    plt.ylabel('Count')
    # Keep original labels: % within insurance = Yes/No
    add_within_category_percent_labels(axIns, df, 'hasInsuranceMapped')
    # Also: share of the whole dataset per insurance category
    add_category_share_of_total_labels(axIns, df, 'hasInsuranceMapped')
    plt.tight_layout()
    plt.savefig(f'{outputDir}/insurance_distribution.png', dpi=300)
    plt.show()

#10) Income-to-poverty ratio (as Percent of Total) 
if 'incomePovertyRatioCapped' in df.columns and 'diabetesStatusMapped' in df.columns:
    plt.figure(figsize=(10, 6))
    try:
        # seaborn >= 0.12
        axIncome = sns.histplot(
            data=df,
            x='incomePovertyRatioCapped',
            hue='diabetesStatusMapped',
            multiple='stack',
            stat='percent',       # bar height = % of total dataset
            common_norm=True,
            palette='cividis',
            bins=20
        )
        plt.ylabel('Percent of Total (%)')
        title_suffix = '(Percent of Total, 60+)'
    except TypeError:
        weights = np.ones(len(df)) * (100.0 / len(df))
        axIncome = sns.histplot(
            data=df,
            x='incomePovertyRatioCapped',
            hue='diabetesStatusMapped',
            multiple='stack',
            weights=weights,
            palette='cividis',
            bins=20
        )
        plt.ylabel('Percent of Total (%)')
        title_suffix = '(Percent of Total via Weights, 60+)'

    plt.title(f'Diabetic vs. Non-Diabetic by Income-to-Poverty Ratio {title_suffix}')
    plt.xlabel('Income-to-Poverty Ratio (Capped at 95th percentile)')

    # Stats annotation (means/medians/SDs)
    statsText = "Income Ratio Stats (60+):\n\n"
    for status, groupDf in df.groupby('diabetesStatusMapped'):
        vals = groupDf['incomePovertyRatioCapped'].dropna()
        if len(vals) == 0:
            continue
        meanVal = vals.mean()
        medianVal = vals.median()
        stdVal = vals.std()
        statsText += f"--- {status} ---\n"
        statsText += f"Mean: {meanVal:.2f}\nMedian: {medianVal:.2f}\nStd Dev: {stdVal:.2f}\n\n"

    plt.text(0.6, 0.95, statsText, transform=axIncome.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{outputDir}/incomePovertyRatio_distribution.png', dpi=300)
    plt.show()

