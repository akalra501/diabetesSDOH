import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# -------- Config --------
dataPath   = "60+/nhanesSeniorCitizens.csv"
targetCol  = "isDiabetic"
testSize   = 0.10
seed       = 42
# ------------------------

# Load data
df = pd.read_csv(dataPath)
if targetCol not in df.columns:
    raise ValueError(f"Target '{targetCol}' not in columns: {list(df.columns)[:10]}")

df = df.dropna(subset=[targetCol])
df = df[df[targetCol].isin([0, 1])]

y = df[targetCol].astype(int)
X = df.drop(columns=[targetCol])

# Feature types
numCols = list(X.select_dtypes(include=np.number).columns)
catCols = list(X.select_dtypes(include=["object", "category"]).columns)

print(f"\nData shape: {df.shape}")
print(f"Positive rate: {y.mean():.3f}")

# Preprocessors
preprocessForLr = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numCols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), catCols)
], remainder="drop")

preprocessForTree = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), numCols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), catCols)
], remainder="drop")

# Train/test split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=testSize, stratify=y, random_state=seed)
print(f"Train size: {len(XTrain)} | Test size: {len(XTest)}")

# Helper to get feature names after pipeline
def getFeatureNames(preprocessor, numCols, catCols):
    numFeats = numCols
    catFeats = list(preprocessor.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(catCols))
    return numFeats + catFeats

# Fit and report
def fitAndReport(modelName, model, preprocessor, XTr, yTr, XTe, yTe, numCols, catCols):
    pipe = ImbPipeline([("pre", preprocessor), ("smote", SMOTE(random_state=seed)), ("clf", model)])
    pipe.fit(XTr, yTr)
    yPred = pipe.predict(XTe)
    yProb = pipe.predict_proba(XTe)[:, 1]

    print(f"\n=== {modelName} + SMOTE ===")
    print("Confusion matrix:\n", confusion_matrix(yTe, yPred))
    print("\nClassification report:")
    print(classification_report(yTe, yPred, target_names=["Non-Diabetic (0)", "Diabetic (1)"]))
    print(f"Accuracy: {accuracy_score(yTe, yPred):.3f} | ROC-AUC: {roc_auc_score(yTe, yProb):.3f} | PR-AUC: {average_precision_score(yTe, yProb):.3f}")

    # Feature importance
    featNames = getFeatureNames(pipe.named_steps["pre"], numCols, catCols)
    clf = pipe.named_steps["clf"]

    if isinstance(clf, LogisticRegression):
        oddsRatios = np.exp(clf.coef_[0])
        orDf = pd.DataFrame({"feature": featNames, "oddsRatio": oddsRatios})
        topOr = orDf.reindex(orDf.oddsRatio.sub(1).abs().sort_values(ascending=False).index).head(10)
        print("\nTop 10 features (LR by |OR-1|):")
        print(topOr.to_string(index=False))

    elif hasattr(clf, "feature_importances_"):
        impDf = pd.DataFrame({"feature": featNames, "importance": clf.feature_importances_})
        topImp = impDf.sort_values("importance", ascending=False).head(10)
        print(f"\nTop 10 features ({modelName} importance):")
        print(topImp.to_string(index=False))

# Models
lrClf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
rfClf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced_subsample")
xgbClf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=seed,
                       n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8)

# Run
fitAndReport("Logistic Regression", lrClf, preprocessForLr, XTrain, yTrain, XTest, yTest, numCols, catCols)
fitAndReport("Random Forest", rfClf, preprocessForTree, XTrain, yTrain, XTest, yTest, numCols, catCols)
fitAndReport("XGBoost", xgbClf, preprocessForTree, XTrain, yTrain, XTest, yTest, numCols, catCols)

print("\nDone.")
