# compare_without_glucose_all_models.py
# Uses seniors file, NO glucose. Compares Base (AgeGroup+Gender) vs +SDOH for LR, RF, XGB.

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, average_precision_score
)

# ----- 1) Load seniors file -----
paths = ["60+/nhanesSeniorCitizens.cv", "60+/nhanesSeniorCitizens.csv"]
df = None
for p in paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"Loaded: {p}")
        break
if df is None:
    raise FileNotFoundError("Could not find 60+/nhanesSeniorCitizens.cv or .csv")

TARGET = "isDiabetic"
expected = [
    "ageGroup","Gender","incomePovertyRatioCapped",
    "educationLevelMapped","raceEthnicityMapped","HasInsurance", TARGET
]
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Keep only what we need; clean target
df = df[expected].copy()
df = df.dropna(subset=[TARGET])
df = df[df[TARGET].isin([0,1])].copy()
y_all = df[TARGET].astype(int)

# Try to coerce HasInsurance to numeric if it's not already
if not pd.api.types.is_numeric_dtype(df["HasInsurance"]):
    df["HasInsurance"] = pd.to_numeric(df["HasInsurance"], errors="coerce")

# Feature sets (NO glucose anywhere)
BASE_FEATS  = ["ageGroup", "Gender"]
SDOH_FEATS  = BASE_FEATS + ["incomePovertyRatioCapped",
                            "educationLevelMapped",
                            "raceEthnicityMapped",
                            "HasInsurance"]

# Restrict to rows complete for the larger (+SDOH) model so we compare apples-to-apples
work = df.dropna(subset=SDOH_FEATS).copy()
y = work[TARGET].astype(int)
X_base = work[BASE_FEATS].copy()
X_sdoh = work[SDOH_FEATS].copy()

print(f"Sample size (complete for +SDOH): {len(work)} | Pos rate: {y.mean():.3f}")

# ----- 2) Train/test split (same indices for both designs) -----
Xb_train, Xb_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.25, stratify=y, random_state=42
)
Xs_train = X_sdoh.loc[Xb_train.index]
Xs_test  = X_sdoh.loc[Xb_test.index]

# ----- 3) Preprocessors -----
def make_ohe(drop_first=True):
    # version-friendly OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first" if drop_first else None, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first" if drop_first else None, sparse=False)

def split_types(X):
    num = list(X.select_dtypes(include=np.number).columns)
    cat = [c for c in X.columns if c not in num]
    return num, cat

def make_preprocessor(X, scale_numeric=False, drop_first=True):
    num_cols, cat_cols = split_types(X)
    steps = []
    if num_cols:
        num_pipe = [("imp", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_pipe.append(("scaler", StandardScaler()))
        steps.append(("num", Pipeline(num_pipe), num_cols))
    if cat_cols:
        steps.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe(drop_first=drop_first))
        ]), cat_cols))
    return ColumnTransformer(steps, remainder="drop"), num_cols, cat_cols

def get_feature_names(pre, num_cols, cat_cols):
    names = []
    # numeric names
    names.extend(num_cols)
    # find the categorical transformer by name
    cat_transformer = None
    for name, transformer, cols in pre.transformers_:
        if name == "cat":
            cat_transformer = transformer
            break
    if cat_transformer is not None and hasattr(cat_transformer.named_steps["ohe"], "get_feature_names_out"):
        ohe = cat_transformer.named_steps["ohe"]
        names.extend(list(ohe.get_feature_names_out(cat_cols)))
    return names

# LR gets scaled numerics; trees do not
pre_base_lr,  num_b_lr,  cat_b_lr  = make_preprocessor(Xb_train, scale_numeric=True,  drop_first=True)
pre_sdoh_lr,  num_s_lr,  cat_s_lr  = make_preprocessor(Xs_train, scale_numeric=True,  drop_first=True)
pre_base_rf,  num_b_rf,  cat_b_rf  = make_preprocessor(Xb_train, scale_numeric=False, drop_first=True)
pre_sdoh_rf,  num_s_rf,  cat_s_rf  = make_preprocessor(Xs_train, scale_numeric=False, drop_first=True)
pre_base_xgb, num_b_xgb, cat_b_xgb = pre_base_rf, num_b_rf, cat_b_rf
pre_sdoh_xgb, num_s_xgb, cat_s_xgb = pre_sdoh_rf, num_s_rf, cat_s_rf

# ----- 4) Models -----
lr_base  = Pipeline([("pre", pre_base_lr),
                     ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])
lr_sdoh  = Pipeline([("pre", pre_sdoh_lr),
                     ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])

rf_base  = Pipeline([("pre", pre_base_rf),
                     ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"))])
rf_sdoh  = Pipeline([("pre", pre_sdoh_rf),
                     ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"))])

# XGBoost optional
try:
    from xgboost import XGBClassifier
    xgb_base = Pipeline([("pre", pre_base_xgb),
                         ("clf", XGBClassifier(
                             objective="binary:logistic", eval_metric="logloss",
                             n_estimators=400, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, random_state=42
                         ))])
    xgb_sdoh = Pipeline([("pre", pre_sdoh_xgb),
                         ("clf", XGBClassifier(
                             objective="binary:logistic", eval_metric="logloss",
                             n_estimators=400, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, random_state=42
                         ))])
    XGB_OK = True
except Exception as e:
    print("\n[Info] xgboost not installed; skipping XGB. Install with: pip install xgboost")
    XGB_OK = False

# ----- 5) Fit & evaluate -----
def fit_eval(name, pipe, Xtr, Xte, ytr, yte, feat_name_fn=None):
    pipe.fit(Xtr, ytr)
    y_pred = pipe.predict(Xte)
    y_prob = pipe.predict_proba(Xte)[:,1]
    print(f"\n=== {name} ===")
    print("Confusion matrix:\n", confusion_matrix(yte, y_pred))
    print("\nClassification report:")
    print(classification_report(yte, y_pred, target_names=["Non-Diabetic (0)","Diabetic (1)"]))
    print(f"Accuracy: {accuracy_score(yte, y_pred):.3f} | "
          f"ROC-AUC: {roc_auc_score(yte, y_prob):.3f} | "
          f"PR-AUC: {average_precision_score(yte, y_prob):.3f}")

    # Top features
    if feat_name_fn is not None:
        feats = feat_name_fn(pipe)
        if hasattr(pipe.named_steps["clf"], "coef_"):  # LR
            coefs = pipe.named_steps["clf"].coef_[0]
            or_df = pd.DataFrame({"feature": feats, "odds_ratio": np.exp(coefs), "abs_coef": np.abs(coefs)})
            top10 = or_df.sort_values("abs_coef", ascending=False).head(10)
            print("\nTop 10 features (LR by |coef|):")
            print(top10[["feature","odds_ratio"]].to_string(index=False))
        elif hasattr(pipe.named_steps["clf"], "feature_importances_"):  # RF/XGB
            imps = pipe.named_steps["clf"].feature_importances_
            imp_df = pd.DataFrame({"feature": feats, "importance": imps}).sort_values("importance", ascending=False).head(10)
            print("\nTop 10 features (by importance):")
            print(imp_df.to_string(index=False))

def feats_base_lr(pipe):  return get_feature_names(pipe.named_steps["pre"], num_b_lr,  cat_b_lr)
def feats_sdoh_lr(pipe):  return get_feature_names(pipe.named_steps["pre"], num_s_lr,  cat_s_lr)
def feats_base_rf(pipe):  return get_feature_names(pipe.named_steps["pre"], num_b_rf,  cat_b_rf)
def feats_sdoh_rf(pipe):  return get_feature_names(pipe.named_steps["pre"], num_s_rf,  cat_s_rf)
def feats_base_xgb(pipe): return get_feature_names(pipe.named_steps["pre"], num_b_xgb, cat_b_xgb)
def feats_sdoh_xgb(pipe): return get_feature_names(pipe.named_steps["pre"], num_s_xgb, cat_s_xgb)

# Logistic Regression
_ = fit_eval("LR Base = ageGroup + Gender", lr_base, Xb_train, Xb_test, y_train, y_test, feats_base_lr)
_ = fit_eval("LR +SDOH (Base + income/education/race/insurance)", lr_sdoh, Xs_train, Xs_test, y_train, y_test, feats_sdoh_lr)

# Random Forest
_ = fit_eval("RF Base = ageGroup + Gender", rf_base, Xb_train, Xb_test, y_train, y_test, feats_base_rf)
_ = fit_eval("RF +SDOH (Base + income/education/race/insurance)", rf_sdoh, Xs_train, Xs_test, y_train, y_test, feats_sdoh_rf)

# XGBoost (if available)
if XGB_OK:
    _ = fit_eval("XGB Base = ageGroup + Gender", xgb_base, Xb_train, Xb_test, y_train, y_test, feats_base_xgb)
    _ = fit_eval("XGB +SDOH (Base + income/education/race/insurance)", xgb_sdoh, Xs_train, Xs_test, y_train, y_test, feats_sdoh_xgb)

print("\nDone.")
