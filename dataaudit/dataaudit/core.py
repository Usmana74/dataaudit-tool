import pandas as pd
import numpy as np
from scipy import stats
from .report import generate_pdf

# ─────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────

def _load(file):
    if isinstance(file, str):
        return pd.read_csv(file)
    elif isinstance(file, pd.DataFrame):
        return file.copy()
    else:
        raise ValueError("Input must be a CSV filepath or a pandas DataFrame.")

def _filename(file):
    if isinstance(file, str):
        return file
    return "dataframe"

# ─────────────────────────────────────────
# INDIVIDUAL FUNCTIONS
# ─────────────────────────────────────────

def summary(file):
    """Print a high-level overview of the dataset."""
    df = _load(file)
    size_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  SUMMARY — {_filename(file)}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Rows        : {df.shape[0]}")
    print(f"  Columns     : {df.shape[1]}")
    print(f"  Memory Usage: {size_kb} KB")
    print(f"  Numeric Cols: {len(df.select_dtypes(include=np.number).columns)}")
    print(f"  Object Cols : {len(df.select_dtypes(include='object').columns)}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return df

def missing(file):
    """Show missing value counts and percentages per column."""
    df = _load(file)
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  MISSING VALUES")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    found = False
    for col in df.columns:
        count = df[col].isnull().sum()
        pct = round((count / len(df)) * 100, 2)
        if count > 0:
            found = True
            if pct >= 50:
                flag = "✖ Drop recommended"
            elif pct >= 10:
                flag = "⚠ Impute recommended"
            else:
                flag = "✔ Safe to impute"
            print(f"  {col:<20} {count} missing ({pct}%)  {flag}")
    if not found:
        print("  ✔ No missing values found.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return df.isnull().sum().to_frame("missing_count")

def duplicates(file):
    """Detect and report duplicate rows."""
    df = _load(file)
    count = df.duplicated().sum()
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  DUPLICATES")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if count == 0:
        print("  ✔ No duplicate rows found.")
    else:
        print(f"  ✖ {count} duplicate rows detected.")
        print(f"  Recommendation: Run df.drop_duplicates() to remove them.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return count

def outliers(file):
    """Detect outliers in numeric columns using the IQR method."""
    df = _load(file)
    numeric = df.select_dtypes(include=np.number)
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  OUTLIERS (IQR Method)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    results = {}
    for col in numeric.columns:
        Q1 = numeric[col].quantile(0.25)
        Q3 = numeric[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        count = ((numeric[col] < lower) | (numeric[col] > upper)).sum()
        results[col] = count
        flag = "⚠ Investigate" if count > 0 else "✔ Clean"
        print(f"  {col:<20} {count} outliers   {flag}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return pd.Series(results, name="outlier_count")

def skewness(file):
    """Show skewness of all numeric columns."""
    df = _load(file)
    numeric = df.select_dtypes(include=np.number)
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  SKEWNESS")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for col in numeric.columns:
        skew = round(numeric[col].skew(), 3)
        if abs(skew) < 0.5:
            label = "✔ Normal"
        elif abs(skew) < 1:
            label = "⚠ Moderate skew"
        else:
            label = "✖ Highly skewed — consider log transform"
        print(f"  {col:<20} {skew:<10} {label}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return numeric.skew().to_frame("skewness")

def cardinality(file):
    """Show unique value counts for categorical columns."""
    df = _load(file)
    cat = df.select_dtypes(include='object')
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  CARDINALITY (Categorical Columns)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for col in cat.columns:
        unique = df[col].nunique()
        pct = round((unique / len(df)) * 100, 1)
        if pct > 80:
            flag = "✖ Likely ID column — drop"
        elif unique > 20:
            flag = "⚠ High cardinality — consider encoding carefully"
        else:
            flag = "✔ Good for encoding"
        print(f"  {col:<20} {unique} unique ({pct}%)   {flag}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return cat.nunique().to_frame("unique_count")

def dtypes(file):
    """Display data types of all columns."""
    df = _load(file)
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  DATA TYPES")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<20} {str(dtype)}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return df.dtypes.to_frame("dtype")

def correlation(file, threshold=0.8):
    """Show highly correlated column pairs above a threshold."""
    df = _load(file)
    numeric = df.select_dtypes(include=np.number)
    corr_matrix = numeric.corr().abs()
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  HIGH CORRELATION (threshold ≥ {threshold})")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    found = False
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and val >= threshold:
                found = True
                print(f"  {idx} ↔ {col}: {round(val, 3)}  ⚠ Consider dropping one")
    if not found:
        print(f"  ✔ No column pairs exceed correlation threshold of {threshold}.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return corr_matrix

def unique_values(file):
    """Show unique value counts for every column."""
    df = _load(file)
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  UNIQUE VALUES PER COLUMN")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for col in df.columns:
        print(f"  {col:<20} {df[col].nunique()} unique values")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return df.nunique().to_frame("unique_count")

def value_counts(file, column):
    """Show value counts for a specific column."""
    df = _load(file)
    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  VALUE COUNTS — {column}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    vc = df[column].value_counts()
    for val, count in vc.items():
        print(f"  {str(val):<25} {count}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return vc

def memory_usage(file):
    """Show memory usage per column."""
    df = _load(file)
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  MEMORY USAGE")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    mem = df.memory_usage(deep=True)
    for col, usage in mem.items():
        print(f"  {str(col):<20} {round(usage / 1024, 3)} KB")
    total = round(mem.sum() / 1024, 2)
    print(f"\n  Total: {total} KB")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return mem

# ─────────────────────────────────────────
# MASTER RUN FUNCTION
# ─────────────────────────────────────────

def run(file):
    """Run a full audit report, then offer PDF download."""
    df = _load(file)
    fname = _filename(file)

    print(f"\n  Analyzing {fname}...\n")

    summary(file)
    missing(file)
    duplicates(file)
    outliers(file)
    skewness(file)
    cardinality(file)
    dtypes(file)
    correlation(file)
    unique_values(file)
    memory_usage(file)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    choice = input("  Download PDF report? (y/n): ").strip().lower()
    if choice == 'y':
        output_name = fname.replace(".csv", "") + "_dataaudit_report.pdf"
        generate_pdf(df, fname, output_name)
        print(f"  ✔ Report saved → {output_name}")
    else:
        print("  PDF export aborted. Audit complete.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")