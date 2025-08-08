import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Load the 3-class engineered dataset.
try:
    # This file contains the Non-Diabetic, Pre-Diabetic, and Diabetic classes.
    modelDf = pd.read_csv('engineeredData.csv')
except FileNotFoundError:
    print("Error: 'engineeredData.csv' not found.")
    exit()

# Separate the data into features (X) and the target variable (y).
X = modelDf.drop('diabetesCategory', axis=1)
y = modelDf['diabetesCategory']

# Identify column types for the pipeline.
numericalFeatures = X.select_dtypes(include=np.number).columns
categoricalFeatures = X.select_dtypes(include=['object', 'category']).columns

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Set up a preprocessor to be used by all models.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numericalFeatures),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categoricalFeatures)
    ], remainder='passthrough')

# Model 1: Logistic Regression (Baseline)
print("\nLogistic Regression:")
logisticPipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
])
logisticPipeline.fit(X_train, y_train)
yPredLr = logisticPipeline.predict(X_test)
print("\nLogistic Regression - Classification Report:")
targetNames = ['Non-Diabetic (0)', 'Pre-Diabetic (1)', 'Diabetic (2)']
print(classification_report(y_test, yPredLr, target_names=targetNames))

# Visualizations for Logistic Regression
cmLr = confusion_matrix(y_test, yPredLr)
plt.figure(figsize=(8, 6))
sns.heatmap(cmLr, annot=True, fmt='d', cmap='Oranges', xticklabels=targetNames, yticklabels=targetNames)
plt.title('Confusion Matrix: Logistic Regression (3-Class)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/confusion_matrix_lr_3_class.png')
plt.show()

# Model 2: Random Forest
print("\nRandom Forest:")
rfPipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=150, class_weight='balanced'))
])
rfPipeline.fit(X_train, y_train)
yPredRf = rfPipeline.predict(X_test)
print("\nRandom Forest - Classification Report:")
print(classification_report(y_test, yPredRf, target_names=targetNames))

# Visualizations for Random Forest
cmRf = confusion_matrix(y_test, yPredRf)
plt.figure(figsize=(8, 6))
sns.heatmap(cmRf, annot=True, fmt='d', cmap='Blues', xticklabels=targetNames, yticklabels=targetNames)
plt.title('Confusion Matrix: Random Forest (3-Class)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/confusion_matrix_rf_3_class.png')
plt.show()

print("\nRandom Forest: Feature Importances")
try:
    oheFeatureNames = rfPipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categoricalFeatures)
    allFeatureNames = np.concatenate([numericalFeatures, oheFeatureNames])
    importances = rfPipeline.named_steps['classifier'].feature_importances_
    featureImportanceDf = pd.DataFrame({'feature': allFeatureNames, 'importance': importances})
    featureImportanceDf = featureImportanceDf.sort_values(by='importance', ascending=False)
    print(featureImportanceDf.head(15))
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=featureImportanceDf.head(15), palette='viridis')
    plt.title('Top 15 Most Important Features (Random Forest, 3-Class)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('plots/feature_importance_rf_3_class.png')
    plt.show()
except Exception as e:
    print(f"Could not generate Random Forest feature importance plot. Error: {e}")


# Model 3: XGBoost
print("\nXGBoost:")
xgbPipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')) # eval_metric is mlogloss for multi-class
])
xgbPipeline.fit(X_train, y_train)
yPredXgb = xgbPipeline.predict(X_test)
print("\nXGBoost - Classification Report:")
print(classification_report(y_test, yPredXgb, target_names=targetNames))

# Visualizations for XGBoost
print("\nGenerating Plots for XGBoost Model")
cmXgb = confusion_matrix(y_test, yPredXgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cmXgb, annot=True, fmt='d', cmap='Greens', xticklabels=targetNames, yticklabels=targetNames)
plt.title('Confusion Matrix: XGBoost (3-Class)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/confusion_matrix_xgboost_3_class.png')
plt.show()

print("\nXGBoost: Feature Importances")
try:
    oheFeatureNames = xgbPipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categoricalFeatures)
    allFeatureNames = np.concatenate([numericalFeatures, oheFeatureNames])
    importances = xgbPipeline.named_steps['classifier'].feature_importances_
    featureImportanceDf = pd.DataFrame({'feature': allFeatureNames, 'importance': importances})
    featureImportanceDf = featureImportanceDf.sort_values(by='importance', ascending=False)
    print(featureImportanceDf.head(15))
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=featureImportanceDf.head(15), palette='inferno')
    plt.title('Top 15 Most Important Features (XGBoost, 3-Class)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('plots/feature_importance_xgboost_3_class.png')
    plt.show()
except Exception as e:
    print(f"Could not generate XGBoost feature importance plot. Error: {e}")
