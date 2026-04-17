# dataaudit

Automated data quality audit library for CSV files and pandas DataFrames.

## Install
pip install dataaudit

## Usage
import dataaudit as da

# Full audit with optional PDF export
da.run("your_data.csv")

# Individual functions
da.summary("your_data.csv")
da.missing("your_data.csv")
da.duplicates("your_data.csv")
da.outliers("your_data.csv")
da.skewness("your_data.csv")
da.cardinality("your_data.csv")
da.dtypes("your_data.csv")
da.correlation("your_data.csv")
da.unique_values("your_data.csv")
da.value_counts("your_data.csv", "column_name")
da.memory_usage("your_data.csv")

# Works with DataFrames too
import pandas as pd
df = pd.read_csv("data.csv")
da.run(df)