# 🐼 Exploratory Data Analysis (EDA) with Pandas — Cheat Sheet

> A comprehensive reference guide for performing EDA using the Pandas library in Python. Covers everything from data loading to advanced transformations, time series, and performance tuning.

---

## 📋 Table of Contents

1. [Data Loading](#1-data-loading)
2. [Basic Data Inspection](#2-basic-data-inspection)
3. [Data Cleaning](#3-data-cleaning)
4. [Data Transformation](#4-data-transformation)
5. [Data Visualization Integration](#5-data-visualization-integration)
6. [Statistical Analysis](#6-statistical-analysis)
7. [Indexing and Selection](#7-indexing-and-selection)
8. [Data Formatting and Conversion](#8-data-formatting-and-conversion)
9. [Advanced Data Transformation](#9-advanced-data-transformation)
10. [Handling Time Series Data](#10-handling-time-series-data)
11. [File Export](#11-file-export)
12. [Data Exploration Techniques](#12-data-exploration-techniques)
13. [Advanced Data Queries](#13-advanced-data-queries)
14. [Memory Optimization](#14-memory-optimization)
15. [Multi-Index Operations](#15-multi-index-operations)
16. [Data Merging Techniques](#16-data-merging-techniques)
17. [Dealing with Duplicates](#17-dealing-with-duplicates)
18. [Custom Operations with Apply](#18-custom-operations-with-apply)
19. [Handling Large Datasets](#19-handling-large-datasets)
20. [Integration with Matplotlib](#20-integration-with-matplotlib-for-custom-plots)
21. [Specialized Data Types Handling](#21-specialized-data-types-handling)
22. [Performance Tuning](#22-performance-tuning)
23. [Visualization Enhancement](#23-visualization-enhancement)
24. [Advanced Grouping and Aggregation](#24-advanced-grouping-and-aggregation)
25. [Time Series Specific Operations](#25-time-series-specific-operations)
26. [Text Data Specific Operations](#26-text-data-specific-operations)
27. [Data Normalization and Standardization](#27-data-normalization-and-standardization)
28. [Working with JSON and XML](#28-working-with-json-and-xml)
29. [Advanced File Handling](#29-advanced-file-handling)
30. [Dealing with Missing Data](#30-dealing-with-missing-data)
31. [Data Reshaping](#31-data-reshaping)
32. [Categorical Data Operations](#32-categorical-data-operations)
33. [Advanced Indexing](#33-advanced-indexing)
34. [Efficient Computations](#34-efficient-computations)
35. [Integration with SciPy and StatsModels](#35-integration-with-scipy-and-statsmodels)

---

## 1. Data Loading

| Operation | Code |
|-----------|------|
| Read CSV File | `df = pd.read_csv('filename.csv')` |
| Read Excel File | `df = pd.read_excel('filename.xlsx')` |
| Read from SQL Database | `df = pd.read_sql(query, connection)` |

---

## 2. Basic Data Inspection

| Operation | Code |
|-----------|------|
| Display Top Rows | `df.head()` |
| Display Bottom Rows | `df.tail()` |
| Display Data Types | `df.dtypes` |
| Summary Statistics | `df.describe()` |
| Display Index, Columns, and Data | `df.info()` |

---

## 3. Data Cleaning

| Operation | Code |
|-----------|------|
| Check for Missing Values | `df.isnull().sum()` |
| Fill Missing Values | `df.fillna(value)` |
| Drop Missing Values | `df.dropna()` |
| Rename Columns | `df.rename(columns={'old_name': 'new_name'})` |
| Drop Columns | `df.drop(columns=['column_name'])` |

---

## 4. Data Transformation

| Operation | Code |
|-----------|------|
| Apply Function | `df['column'].apply(lambda x: function(x))` |
| Group By and Aggregate | `df.groupby('column').agg({'column': 'sum'})` |
| Pivot Tables | `df.pivot_table(index='column1', values='column2', aggfunc='mean')` |
| Merge DataFrames | `pd.merge(df1, df2, on='column')` |
| Concatenate DataFrames | `pd.concat([df1, df2])` |

---

## 5. Data Visualization Integration

| Operation | Code |
|-----------|------|
| Histogram | `df['column'].hist()` |
| Boxplot | `df.boxplot(column=['column1', 'column2'])` |
| Scatter Plot | `df.plot.scatter(x='col1', y='col2')` |
| Line Plot | `df.plot.line()` |
| Bar Chart | `df['column'].value_counts().plot.bar()` |

---

## 6. Statistical Analysis

| Operation | Code |
|-----------|------|
| Correlation Matrix | `df.corr()` |
| Covariance Matrix | `df.cov()` |
| Value Counts | `df['column'].value_counts()` |
| Unique Values in Column | `df['column'].unique()` |
| Number of Unique Values | `df['column'].nunique()` |

---

## 7. Indexing and Selection

| Operation | Code |
|-----------|------|
| Select Column | `df['column']` |
| Select Multiple Columns | `df[['col1', 'col2']]` |
| Select Rows by Position | `df.iloc[0:5]` |
| Select Rows by Label | `df.loc[0:5]` |
| Conditional Selection | `df[df['column'] > value]` |

---

## 8. Data Formatting and Conversion

| Operation | Code |
|-----------|------|
| Convert Data Types | `df['column'].astype('type')` |
| String Operations | `df['column'].str.lower()` |
| Datetime Conversion | `pd.to_datetime(df['column'])` |
| Setting Index | `df.set_index('column')` |

---

## 9. Advanced Data Transformation

| Operation | Code |
|-----------|------|
| Lambda Functions | `df.apply(lambda x: x + 1)` |
| Pivot Longer/Wider Format | `df.melt(id_vars=['col1'])` |
| Stack/Unstack | `df.stack()`, `df.unstack()` |
| Cross Tabulations | `pd.crosstab(df['col1'], df['col2'])` |

---

## 10. Handling Time Series Data

| Operation | Code |
|-----------|------|
| Set Datetime Index | `df.set_index(pd.to_datetime(df['date']))` |
| Resampling Data | `df.resample('M').mean()` |
| Rolling Window Operations | `df.rolling(window=5).mean()` |

---

## 11. File Export

| Operation | Code |
|-----------|------|
| Write to CSV | `df.to_csv('filename.csv')` |
| Write to Excel | `df.to_excel('filename.xlsx')` |
| Write to SQL Database | `df.to_sql('table_name', connection)` |

---

## 12. Data Exploration Techniques

| Operation | Code |
|-----------|------|
| Profile Report | `from pandas_profiling import ProfileReport; ProfileReport(df)` |
| Pairplot (with seaborn) | `import seaborn as sns; sns.pairplot(df)` |
| Heatmap for Correlation | `sns.heatmap(df.corr(), annot=True)` |

---

## 13. Advanced Data Queries

| Operation | Code |
|-----------|------|
| Query Function | `df.query('column > value')` |
| Filtering with isin | `df[df['column'].isin([value1, value2])]` |

---

## 14. Memory Optimization

| Operation | Code |
|-----------|------|
| Reducing Memory Usage | `df.memory_usage(deep=True)` |
| Change Data Types to Save Memory | `df['column'].astype('category')` |

---

## 15. Multi-Index Operations

| Operation | Code |
|-----------|------|
| Creating MultiIndex | `df.set_index(['col1', 'col2'])` |
| Slicing on MultiIndex | `df.loc[(slice('index1_start', 'index1_end'), slice('index2_start', 'index2_end'))]` |

---

## 16. Data Merging Techniques

| Operation | Code |
|-----------|------|
| Outer Join | `pd.merge(df1, df2, on='column', how='outer')` |
| Inner Join | `pd.merge(df1, df2, on='column', how='inner')` |
| Left Join | `pd.merge(df1, df2, on='column', how='left')` |
| Right Join | `pd.merge(df1, df2, on='column', how='right')` |

---

## 17. Dealing with Duplicates

| Operation | Code |
|-----------|------|
| Finding Duplicates | `df.duplicated()` |
| Removing Duplicates | `df.drop_duplicates()` |

---

## 18. Custom Operations with Apply

| Operation | Code |
|-----------|------|
| Custom Apply Functions | `df.apply(lambda row: custom_func(row['col1'], row['col2']), axis=1)` |

---

## 19. Handling Large Datasets

| Operation | Code |
|-----------|------|
| Chunking Large Files | `pd.read_csv('large_file.csv', chunksize=1000)` |
| Iterating Through Data Chunks | `for chunk in pd.read_csv('file.csv', chunksize=500): process(chunk)` |

---

## 20. Integration with Matplotlib for Custom Plots

| Operation | Code |
|-----------|------|
| Custom Plotting | `import matplotlib.pyplot as plt; df.plot(); plt.show()` |

---

## 21. Specialized Data Types Handling

| Operation | Code |
|-----------|------|
| Working with Categorical Data | `df['column'].astype('category')` |
| Dealing with Sparse Data | `pd.arrays.SparseArray(df['column'])` |

---

## 22. Performance Tuning

| Operation | Code |
|-----------|------|
| Using Swifter for Faster Apply | `import swifter; df['column'].swifter.apply(lambda x: func(x))` |
| Parallel Processing with Dask | `import dask.dataframe as dd; ddf = dd.from_pandas(df, npartitions=10)` |

---

## 23. Visualization Enhancement

| Operation | Code |
|-----------|------|
| Customize Plot Style | `plt.style.use('ggplot')` |
| Histogram with Bins Specification | `df['column'].hist(bins=20)` |
| Boxplot Grouped by Category | `df.boxplot(column='num_column', by='cat_column')` |

---

## 24. Advanced Grouping and Aggregation

| Operation | Code |
|-----------|------|
| Group by Multiple Columns | `df.groupby(['col1', 'col2']).mean()` |
| Aggregate with Multiple Functions | `df.groupby('col').agg(['mean', 'sum'])` |
| Transform Function | `df.groupby('col').transform(lambda x: x - x.mean())` |

---

## 25. Time Series Specific Operations

| Operation | Code |
|-----------|------|
| Time-Based Grouping | `df.groupby(pd.Grouper(key='date_col', freq='M')).sum()` |
| Shifting Series for Lag Analysis | `df['column'].shift(1)` |
| Resample Time Series Data | `df.resample('M', on='date_col').mean()` |

---

## 26. Text Data Specific Operations

| Operation | Code |
|-----------|------|
| String Contains | `df[df['column'].str.contains('substring')]` |
| String Split | `df['column'].str.split(' ', expand=True)` |
| Regular Expression Extraction | `df['column'].str.extract(r'(regex)')` |

---

## 27. Data Normalization and Standardization

| Operation | Code |
|-----------|------|
| Min-Max Normalization | `(df['column'] - df['column'].min()) / (df['column'].max() - df['column'].min())` |
| Z-Score Standardization | `(df['column'] - df['column'].mean()) / df['column'].std()` |

---

## 28. Working with JSON and XML

| Operation | Code |
|-----------|------|
| Reading JSON | `df = pd.read_json('filename.json')` |
| Reading XML | `df = pd.read_xml('filename.xml')` |

---

## 29. Advanced File Handling

| Operation | Code |
|-----------|------|
| Read CSV with Specific Delimiter | `df = pd.read_csv('filename.csv', delimiter=';')` |
| Writing to JSON | `df.to_json('filename.json')` |

---

## 30. Dealing with Missing Data

| Operation | Code |
|-----------|------|
| Interpolate Missing Values | `df['column'].interpolate()` |
| Forward Fill Missing Values | `df['column'].ffill()` |
| Backward Fill Missing Values | `df['column'].bfill()` |

---

## 31. Data Reshaping

| Operation | Code |
|-----------|------|
| Wide to Long Format | `pd.wide_to_long(df, ['col'], i='id_col', j='year')` |
| Long to Wide Format | `df.pivot(index='id_col', columns='year', values='col')` |

---

## 32. Categorical Data Operations

| Operation | Code |
|-----------|------|
| Convert Column to Categorical | `df['column'] = df['column'].astype('category')` |
| Order Categories | `df['column'].cat.set_categories(['cat1', 'cat2'], ordered=True)` |

---

## 33. Advanced Indexing

| Operation | Code |
|-----------|------|
| Reset Index | `df.reset_index(drop=True)` |
| Set Multiple Indexes | `df.set_index(['col1', 'col2'])` |
| MultiIndex Slicing | `df.xs(key='value', level='level_name')` |

---

## 34. Efficient Computations

| Operation | Code |
|-----------|------|
| Use of eval() for Efficient Operations | `df.eval('col1 + col2')` |
| Query Method for Filtering | `df.query('col1 < col2')` |

---

## 35. Integration with SciPy and StatsModels

| Operation | Code |
|-----------|------|
| Linear Regression (with statsmodels) | `import statsmodels.api as sm; sm.OLS(y, X).fit()` |
| Kurtosis and Skewness (with SciPy) | `from scipy.stats import kurtosis, skew; kurtosis(df['column']), skew(df['column'])` |

---

## 🚀 Getting Started

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 📚 Resources

- [Pandas Official Documentation](https://pandas.pydata.org/docs/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
- [StatsModels Documentation](https://www.statsmodels.org/stable/index.html)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a pull request or submit an issue.

---

## ⭐ Show your support

If this cheat sheet helped you, please give it a ⭐ on GitHub!

---

*Happy Analyzing! 🐼📊*
