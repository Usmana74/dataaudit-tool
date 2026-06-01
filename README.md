# dataaudit

> **Automated ML dataset auditing in a single function call.**

[![PyPI version](https://img.shields.io/pypi/v/dataaudit-tool.svg)](https://pypi.org/project/dataaudit-tool/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI Downloads](https://img.shields.io/pypi/dm/dataaudit-tool.svg)](https://pypi.org/project/dataaudit-tool/)

`dataaudit` replaces 10–15 manual EDA inspection steps with a single function call. Pass any CSV filepath or pandas DataFrame — get a complete data quality report covering missing values, outliers, skewness, cardinality, correlation, memory usage, and more. Optionally export the full report as a styled PDF.

Built for ML practitioners who want actionable pre-processing diagnostics fast.

---

## Install

```bash
pip install dataaudit-tool
```

---

## Quickstart

```python
import dataaudit as da

# Full audit — CSV file
da.run("your_data.csv")

# Full audit — pandas DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
da.run(df)
```

Output includes shape, missing values with imputation recommendations, duplicate detection, IQR-based outlier detection, skewness flags, cardinality analysis, high-correlation pairs, and memory usage — all with colour-coded actionability flags.

At the end, you are prompted to export the full report as a **PDF** (`_dataaudit_report.pdf`).

---

## Full API

All functions accept either a **CSV filepath** (string) or a **pandas DataFrame**.

```python
import dataaudit as da

da.summary(file)                        # Shape, memory, column type breakdown
da.missing(file)                        # Missing counts + imputation recommendations
da.duplicates(file)                     # Duplicate row detection
da.outliers(file)                       # IQR-based outlier detection per numeric column
da.skewness(file)                       # Skewness with log-transform recommendations
da.cardinality(file)                    # Unique value counts for categorical columns
da.dtypes(file)                         # Data types for all columns
da.correlation(file, threshold=0.8)     # Highly correlated column pairs
da.unique_values(file)                  # Unique value counts for all columns
da.value_counts(file, "column_name")    # Value frequency for a specific column
da.memory_usage(file)                   # Per-column memory footprint

da.run(file)                            # Run all of the above + optional PDF export
```

---

## Example Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SUMMARY — titanic.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rows        : 891
  Columns     : 12
  Memory Usage: 83.6 KB
  Numeric Cols: 7
  Object Cols : 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MISSING VALUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Age                  177 missing (19.87%)  ⚠ Impute recommended
  Cabin                687 missing (77.1%)   ✖ Drop recommended
  Embarked             2 missing (0.22%)     ✔ Safe to impute
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OUTLIERS (IQR Method)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Age                  11 outliers    ⚠ Investigate
  Fare                 116 outliers   ⚠ Investigate
  SibSp                46 outliers    ⚠ Investigate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## PDF Report

Calling `da.run(file)` prompts you to export a styled PDF report with:
- Dark-header branding, colour-coded flags (green / amber / red)
- All 9 audit sections in sequence
- Per-column memory breakdown
- Auto page breaks, footer with page numbers

---

## Repository Structure

```
dataaudit/
├── dataaudit/
│   ├── __init__.py       # Public API exports
│   ├── core.py           # All audit functions
│   └── report.py         # PDF generation (fpdf2)
├── pyproject.toml        # PyPI packaging config
├── README.md
└── LICENSE
```

---

## Dependencies

```
pandas>=1.3
numpy>=1.21
scipy>=1.7
fpdf2>=2.7
```

---

## Motivation

Every ML project starts the same way: load the data, manually check shape, count nulls, look for duplicates, scan for outliers, check correlations. This is 15–20 lines of boilerplate that gets rewritten from scratch every time. `dataaudit` collapses that into one call with opinionated, actionable output — so you spend time on modelling, not on setup.

---

## Author

**Mohammad Usman Ahmad**
BS Computer Science (AI/CV), PMAS Arid Agriculture University, Pakistan · CGPA: 3.8/4.0

[GitHub](https://github.com/Usmana74) · [LinkedIn](https://linkedin.com/in/usman-ahmad-297b63262) · [PyPI](https://pypi.org/project/dataaudit-tool/)
