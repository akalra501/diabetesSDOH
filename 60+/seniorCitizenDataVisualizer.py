import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("nhanesSeniorCitizens.csv")

# 1. Print missing value count and non-missing count for each column
print("Missing values and counts per column:\n")
for col in df.columns:
    n_missing = df[col].isna().sum()
    n_non_missing = df[col].notna().sum()
    print(f"{col}: Missing = {n_missing}, Non-missing = {n_non_missing}")

# 2. Visualize each variable
for col in df.columns:
    plt.figure(figsize=(6, 4))
    if pd.api.types.is_numeric_dtype(df[col]):
        plt.hist(df[col].dropna(), bins=30, edgecolor='black')
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")
    else:
        df[col].value_counts(dropna=False).plot(kind='bar', edgecolor='black')
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Counts for {col}")
    plt.tight_layout()
    plt.show()