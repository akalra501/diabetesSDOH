import pandas as pd

def missing_percent_table(filepath, group_col=None):
    """
    Calculate missing percentage for each column in a CSV.
    Optionally calculate missing percentage within each group (group_col).
    """
    df = pd.read_csv(filepath)
    n_total = len(df)

    overall = pd.DataFrame({
        'column': df.columns,
        'dtype': [str(df[col].dtype) for col in df.columns],
        'n_rows': n_total,
        'n_missing': df.isna().sum().values,
        'missing_pct': (df.isna().sum() / n_total * 100).round(2),
        'n_unique': [df[col].nunique(dropna=True) for col in df.columns]
    }).sort_values('missing_pct', ascending=False).reset_index(drop=True)

    if group_col:
        if group_col not in df.columns:
            raise ValueError(f"{group_col} not found in DataFrame")
        by_group = []
        for group, subset in df.groupby(group_col, dropna=False):
            n_group = len(subset)
            for col in df.columns:
                n_missing = subset[col].isna().sum()
                pct_missing = (n_missing / n_group * 100).round(2) if n_group > 0 else 0
                by_group.append({
                    'group': group,
                    'column': col,
                    'n_rows': n_group,
                    'n_missing': n_missing,
                    'missing_pct': pct_missing
                })
        by_group_df = pd.DataFrame(by_group).sort_values(['column', 'group']).reset_index(drop=True)
        return overall, by_group_df

    return overall

# Example: print overall missingness
overall_missing = missing_percent_table('nhanesMerge1999_2023_REVISED.csv')
print(overall_missing)

# Example with grouping:
# overall, by_cycle = missing_percent_table('nhanesMerge1999_2023.csv', group_col='SurveyCycle')
# print(overall)
# print(by_cycle)
