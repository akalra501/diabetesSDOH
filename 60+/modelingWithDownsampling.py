# ===== Hybrid Resampling with SMOTE + Downsampling =====

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

# -------- Config --------
dataPath = "60+/nhanesSeniorCitizens.csv"
targetCol = "isDiabetic"
testSize = 0.10
seed = 42
undersampleFraction = 0.2
 # Keep only 50% of non-diabetic cases before SMOTE
# ------------------------

# 1) Load data
df = pd.read_csv(dataPath)
if targetCol not in df.columns:
    raise ValueError(f"Target '{targetCol}' not in columns: {list(df.columns)[:12]} ...")

df = df.dropna(subset=[targetCol])
df = df[df[targetCol].isin([0, 1])]

y = df[targetCol].astype(int)
X = df.drop(columns=[targetCol])

# Feature types
numCols = list(X.select_dtypes(include=np.number).columns)
catCols = list(X.select_dtypes(include=["object", "category"]).columns)

print(f"\nData shape: {df.shape}")
print(f"Positive rate: {y.mean():.3f}")

# 2) Undersample non-diabetics
dfNonDiabetic = df[df[targetCol] == 0]
dfDiabetic = df[df[targetCol] == 1]

dfNonDiabeticSample = dfNonDiabetic.sample(frac=undersampleFraction, random_state=seed)
dfHybrid = pd.concat([dfNonDiabeticSample, dfDiabetic], axis=0).sample(frac=1, random_state=seed)

y = dfHybrid[targetCol].astype(int)
X = dfHybrid.drop(columns=[targetCol])

print(f"After undersampling: {X.shape}, positive rate = {y.mean():.3f}")

# 3) Train/test split
XTrain, XTest, yTrain, yTest = train_test_split(
    X, y, test_size=testSize, stratify=y, random_state=seed
)

# 4) Preprocessing
preprocessLR = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numCols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), catCols),
    ],
    remainder="drop"
)

preprocessTree = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), numCols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), catCols),
    ],
    remainder="drop"
)

# 5) Models with SMOTE
def buildPipeline(preprocessor, model):
    return ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=seed)),
        ("model", model)
    ])

lrModel = buildPipeline(preprocessLR, LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
rfModel = buildPipeline(preprocessTree, RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced_subsample"))
xgbModel = buildPipeline(preprocessTree, XGBClassifier(
    objective="binary:logistic", eval_metric="logloss", use_label_encoder=False,
    n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=seed
))

# 6) Fit & Report
def fitAndReport(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    yPred = model.predict(Xte)
    yProb = model.predict_proba(Xte)[:, 1]

    print(f"\n=== {name} ===")
    print("Confusion Matrix:\n", confusion_matrix(yte, yPred))
    print("\nClassification Report:")
    print(classification_report(yte, yPred, target_names=["Non-Diabetic (0)", "Diabetic (1)"]))
    print(f"Accuracy: {accuracy_score(yte, yPred):.3f} | ROC-AUC: {roc_auc_score(yte, yProb):.3f} | PR-AUC: {average_precision_score(yte, yProb):.3f}")

    # Feature importance / odds ratios
    if name.lower().startswith("logistic"):
        try:
            featureNames = model.named_steps["preprocess"].transformers_[0][2] + \
                           list(model.named_steps["preprocess"].named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(catCols))
            coefs = model.named_steps["model"].coef_.ravel()
            oddsRatios = pd.DataFrame({"feature": featureNames, "oddsRatio": np.exp(coefs)})
            print("\nTop features (by |OR-1|):")
            print(oddsRatios.reindex(oddsRatios.oddsRatio.sub(1).abs().sort_values(ascending=False).index).head(10))
        except Exception as e:
            print(f"Could not extract logistic regression features: {e}")
    else:
        try:
            featureNames = model.named_steps["preprocess"].transformers_[0][2] + \
                           list(model.named_steps["preprocess"].named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(catCols))
            if hasattr(model.named_steps["model"], "feature_importances_"):
                importances = model.named_steps["model"].feature_importances_
                impDf = pd.DataFrame({"feature": featureNames, "importance": importances})
                print("\nTop features (by importance):")
                print(impDf.sort_values("importance", ascending=False).head(10))
        except Exception as e:
            print(f"Could not extract feature importances: {e}")

# Run all
fitAndReport("Logistic Regression + Hybrid Resampling", lrModel, XTrain, yTrain, XTest, yTest)
fitAndReport("Random Forest + Hybrid Resampling", rfModel, XTrain, yTrain, XTest, yTest)
fitAndReport("XGBoost + Hybrid Resampling", xgbModel, XTrain, yTrain, XTest, yTest)

print("\nDone.")
