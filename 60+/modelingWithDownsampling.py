# ===== Hybrid Resampling with SMOTE + Downsampling =====
# This script trains three classifiers (Logistic Regression, Random Forest, XGBoost)
# on a diabetes label using a hybrid resampling approach:
#   1) Downsample the majority class (non-diabetics) to reduce imbalance
#   2) Apply SMOTE within the pipeline (after preprocessing) to oversample the minority class
# It then prints evaluation metrics and plots ROC and PR curves.

import numpy as np
import pandas as pd

# Model selection & preprocessing utilities
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, accuracy_score
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# Plotting
import os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score


#Setting important variables.
dataPath = "60+/nhanesSeniorCitizens.csv"
targetCol = "isDiabetic"
testSize = 0.15
seed = 42
undersampleFraction = 0.2


# 1) Load data
df = pd.read_csv(dataPath)
if targetCol not in df.columns:
    raise ValueError(f"Target '{targetCol}' not in columns: {list(df.columns)[:12]} ...")

# Ensure clean binary target (0/1) and no missing labels
df = df.dropna(subset=[targetCol])
df = df[df[targetCol].isin([0, 1])]

y = df[targetCol].astype(int)
X = df.drop(columns=[targetCol])

# Feature types (numeric vs categorical) for column-wise preprocessing
numCols = list(X.select_dtypes(include=np.number).columns)
catCols = list(X.select_dtypes(include=["object", "category"]).columns)

print(f"\nData shape: {df.shape}")
print(f"Positive rate: {y.mean():.3f}")


# 2) Undersample non-diabetics
# Downsample majority class to reduce class imbalance before SMOTE.
dfNonDiabetic = df[df[targetCol] == 0]
dfDiabetic = df[df[targetCol] == 1]

dfNonDiabeticSample = dfNonDiabetic.sample(frac=undersampleFraction, random_state=seed)
dfHybrid = pd.concat([dfNonDiabeticSample, dfDiabetic], axis=0).sample(frac=1, random_state=seed)

y = dfHybrid[targetCol].astype(int)
X = dfHybrid.drop(columns=[targetCol])

print(f"After undersampling: {X.shape}, positive rate = {y.mean():.3f}")


# 3) Train/test split
# Stratify to preserve class proportions in train and test splits.
XTrain, XTest, yTrain, yTest = train_test_split(
    X, y, test_size=testSize, stratify=y, random_state=seed
)
print(f"\nRequested test_size fraction: {testSize}")
print(f"Train size: {XTrain.shape[0]} | Test size: {XTest.shape[0]}")
print(f"Train class counts: {yTrain.value_counts().to_dict()}")
print(f"Test  class counts: {yTest.value_counts().to_dict()}")
print(f"Test prevalence (diabetic=1): {yTest.mean():.3f}")


# 4) Preprocessing
# Two ColumnTransformers:
#   - preprocessLR: numeric median-impute + standardize; categorical impute + OHE
#   - preprocessTree: numeric median-impute; categorical impute + OHE (no scaling for trees)

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
    """
    Build an imbalanced-learn pipeline with:
      preprocess (impute/encode/scale) -> SMOTE -> model
    SMOTE occurs after preprocessing so it works in fully numeric space.
    """
    return ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=seed)),
        ("model", model)
    ])

# Define three model pipelines with identical random seeds for reproducibility
lrModel = buildPipeline(preprocessLR, LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
rfModel = buildPipeline(preprocessTree, RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced_subsample"))
xgbModel = buildPipeline(preprocessTree, XGBClassifier(
    objective="binary:logistic", eval_metric="logloss", use_label_encoder=False,
    n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=seed
))


# 6) Fit & Report
def fitAndReport(name, model, Xtr, ytr, Xte, yte):
    """
    Fit a model, generate predictions on the test set, and print:
      - Confusion matrix
      - Classification report
      - Accuracy, ROC-AUC, PR-AUC
    Also attempts to show:
      - Odds ratios for Logistic Regression
      - Feature importances for tree-based models (if available)
    """
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


#ROC & PR plotting for fitted models
def _get_scores(clf, X):
    """Return positive-class scores for ROC/PR. Falls back to decision_function if needed."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        # scale to [0,1] so AP behaves
        s_min, s_max = np.min(s), np.max(s)
        return (s - s_min) / (s_max - s_min + 1e-12)
    else:
        raise ValueError(f"Model {type(clf).__name__} has neither predict_proba nor decision_function.")


def plot_roc_pr(models_dict, X_test, y_test, out_dir="60+plots/metrics"):
    """
    Plot ROC and Precision–Recall curves for multiple fitted models on the held-out test set.
    Saves two PNG files to `out_dir` and prints a compact AUC summary.
      - models_dict: dict like {"LR": lrModel, "RF": rfModel, "XGB": xgbModel}
      - X_test, y_test: held-out test features/labels
    """
    os.makedirs(out_dir, exist_ok=True)
    prevalence = y_test.mean()

    # Plotting ROC
    plt.figure(figsize=(7, 6))
    for name, model in models_dict.items():
        scores = _get_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], ls="--", c="gray", lw=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves (60+ test set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.show()
    print(f"Saved: {roc_path}")

    # Plotting PR-AUC
    plt.figure(figsize=(7, 6))
    for name, model in models_dict.items():
        scores = _get_scores(model, X_test)
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        plt.plot(rec, prec, label=f"{name} (PR-AUC={ap:.3f})")
   
    # Baseline = prevalence
    plt.hlines(prevalence, 0, 1, colors="gray", linestyles="--", label=f"Baseline (π={prevalence:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (60+ test set)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(out_dir, "pr_curves.png")
    plt.savefig(pr_path, dpi=300)
    plt.show()
    print(f"Saved: {pr_path}")

    # Summary Table
    print("\nAUC Summary (test set): ")
    for name, model in models_dict.items():
        s = _get_scores(model, X_test)
        print(f"{name:>8} | ROC-AUC: {roc_auc_score(y_test, s):.3f} | PR-AUC: {average_precision_score(y_test, s):.3f}")


# Usage
try:
    models = {
        "Logistic": lrModel,
        "RandomForest": rfModel,
        "XGBoost": xgbModel
    }
    plot_roc_pr(models, XTest, yTest, out_dir="60+plots/metrics")
except NameError as e:
    print("Make sure lrModel, rfModel, xgbModel are fitted and XTest/yTest exist before calling plot_roc_pr().")
    print("Error:", e)
