import scipy.stats as stats
from scipy.stats import ttest_ind
import pandas as pd

df = pd.read_csv('60+/nhanesSeniorCitizens.csv')
contingency = pd.crosstab(df['educationLevelMapped'], df['isDiabetic'])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-square p-value: {p}")


diabetic = df[df['isDiabetic'] == 1]['incomePovertyRatioCapped']
nondiabetic = df[df['isDiabetic'] == 0]['incomePovertyRatioCapped']
t_stat, p_val = ttest_ind(diabetic.dropna(), nondiabetic.dropna())
print(f"T-test p-value: {p_val}")
