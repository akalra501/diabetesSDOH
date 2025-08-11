import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Cleaning the dataset
df = pd.read_csv("engineeredBinaryFeatures_REVISED.csv")
df2 = df[df['ageGroup'] == "Senior_60+"]
print(len(df2))
df2.to_csv('nhanesSeniorCitizens.csv', index=False)

# dfCleaned = df.dropna()
# dfCleaned = dfCleaned[dfCleaned['EducationLevel'] < 6.0]
# dfCleaned.to_csv("nhanesMerge1999_2023_cleaned.csv", index=False)

# dfUnder80 = dfCleaned[dfCleaned['Age'] < 80]
# dfOver80 = dfCleaned[dfCleaned['Age'] > 80]
# print('People over 80:', len(dfOver80))

# dfDiabetes = dfUnder80[dfUnder80['Diabetes'] == 1]
# dfNoDiabetes = dfUnder80[dfUnder80['Diabetes'] == 0]

# os.makedirs('plots', exist_ok=True)

# # Helper: Add stats to subplot
# def addStats(ax, data, variable):
#     meanVal = data[variable].mean()
#     medianVal = data[variable].median()
#     stdVal = data[variable].std()
#     count = data[variable].count()
    
#     ax.text(
#         0.95, 0.95,
#         f"count = {count}\nmean = {meanVal:.2f}\nstdev = {stdVal:.2f}\nmedian = {medianVal:.2f}",
#         transform=ax.transAxes,
#         ha='right',
#         va='top',
#         fontsize=8,
#         bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.7)
#     )

# # -------------------------
# # Function to make side-by-side plots
# # -------------------------
# def makePlots(variable, plotType):
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

#     if plotType == 'hist':
#         sns.histplot(dfDiabetes[variable], ax=axs[0], color='red', bins=30)
#         sns.histplot(dfNoDiabetes[variable], ax=axs[1], color='green', bins=30)
#     elif plotType == 'count':
#         sns.countplot(x=variable, data=dfDiabetes, ax=axs[0], color='red')
#         sns.countplot(x=variable, data=dfNoDiabetes, ax=axs[1], color='green')
#     else:
#         raise ValueError('Enter correct plot type (hist or count)')

#     axs[0].set_title(f"{variable} (Diabetic Group)", fontsize=12)
#     axs[1].set_title(f"{variable} (Non-Diabetic Group)", fontsize=12)
#     axs[0].set_xlabel(variable, fontsize=8)
#     axs[1].set_xlabel(variable, fontsize=8)
#     axs[0].set_ylabel("Count", fontsize=8)
#     axs[1].set_ylabel("Count", fontsize=8)

#     for ax, subset in zip(axs, [dfDiabetes, dfNoDiabetes]):
#         addStats(ax, subset, variable)
#         ax.tick_params(axis='x', labelsize=8)
#         ax.tick_params(axis='y', labelsize=8)

#     plt.tight_layout()
#     plt.savefig(f"plots/{variable}Distribution.png")
#     plt.close()

# # Making plots
# makePlots('Age', 'hist')
# makePlots('IncomePovertyRatio', 'hist')
# makePlots('FastingGlucose', 'hist')
# makePlots('EducationLevel', 'count')
# makePlots('HasInsurance', 'count')



# Class distribution
# print(df_cleaned['Diabetes'].value_counts())
# print(df_cleaned['HasInsurance'].value_counts())
# insuredPeople = df_cleaned[df_cleaned["HasInsurance"] == 1]
# print(insuredPeople['Diabetes'].value_counts())

#visualizations:
# fig, axs = plt.subplots(2,3, figsize=( 15,10))

# #age visualization
# sns.histplot(dfUnder80['Age'], bins = 30, kde = False, ax=axs[0,0])
# axs[0,0].set_title('Age Distribution', fontsize = 10)
# axs[0,0].set_xlabel('Age', fontsize = 6)

# #income poverty ratio visualization
# sns.histplot(dfUnder80['IncomePovertyRatio'], bins = 10, ax=axs[0,1])
# axs[0,1].set_title('Income poverty ratio distribution', fontsize = 10)
# axs[0,1].set_xlabel('Income Poverty Ratio', fontsize = 6)

# #education level visualization
# sns.countplot(x = 'EducationLevel', data = dfUnder80, ax=axs[0,2])
# axs[0,2].set_title('education level', fontsize = 10)
# axs[0,2].set_xticks([1,2,3,4,5])
# axs[0,2].set_xlabel('Education level', fontsize = 6)

# #fasting glucose visualization.
# sns.boxplot(x = 'Diabetes', y = 'FastingGlucose', data = dfUnder80, ax=axs[1,1])
# axs[1,1].set_title('distribution of Diabetes based on fasting glucose', fontsize = 10)
# axs[1,1].set_xlabel('Diabetes', fontsize = 6)

# #insurance status visualization.
# sns.countplot(x='HasInsurance', data=dfUnder80, ax=axs[1,0])
# axs[1,0].set_title('Distribution of Insurance status of individuals', fontsize = 10)
# axs[1,0].set_xlabel('Insurance status', fontsize = 6)

# #visualization of people with glucose of over 300
# sns.countplot(x='HasInsurance', data = dfDiabetes, ax=axs[1,2])
# axs[1,2].set_title("Glucose over 160", fontsize = 10)
# axs[1,2].set_xlabel('Insurance Status', fontsize = 6)

# # plt.show()
# print('Total people in the dataset: ', len(dfUnder80))
# print("People with fasting glucose over 160: "  , len(dfDiabetes))
# print(dfDiabetes["HasInsurance"].value_counts())

# crossTabulation1 = pd.crosstab(dfUnder80['Diabetes'], dfUnder80['HasInsurance'], normalize='index')
# crossTabulation2 = pd.crosstab(dfUnder80['Diabetes'], dfUnder80['EducationLevel'], normalize='index')
# print(crossTabulation1)
# print(crossTabulation2)

# plt.figure(figsize=(8, 6))  # New figure
# sns.heatmap(dfUnder80.corr(numeric_only=True), annot=True, cmap='turbo')
# plt.title("Correlation Heatmap")
# plt.tight_layout()
# # plt.show()

# #Grouped Statistics
# diabetesRateByInsuranceStatus = dfUnder80.groupby('HasInsurance')['Diabetes'].value_counts(normalize=True)
# diabetesRateByEducationLevel = (dfUnder80.groupby('EducationLevel')['Diabetes'].mean())
# print(dfUnder80.groupby(['HasInsurance', 'Diabetes'])['FastingGlucose'].mean())

# plt.figure(figsize=(8,5))
# diabetesRateByEducationLevel.plot(kind='bar')
# plt.title("Diabetes Rate by Education Level", fontsize=14)
# plt.xlabel("Education Level", fontsize=12)
# plt.ylabel("Diabetes Rate", fontsize=12)
# plt.ylim(0, 1)
# plt.tight_layout()
# plt.show()