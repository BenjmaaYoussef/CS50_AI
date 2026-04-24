# DataCo Supply Chain Preprocessing — Complete Explanation

## Overview

This notebook preprocesses supply chain data to predict late deliveries. **Goal**: Build a clean, leakage-free dataset ready for machine learning models. The dataset has ~180K orders, and we're predicting whether an order will be late (binary classification).

---

## Table of Contents
1. [Key Concepts](#key-concepts)
2. [Section 1: Project Overview](#section-1-project-overview)
3. [Section 2: Data Loading & Audit](#section-2-data-loading--initial-audit)
4. [Section 3: Inference Holdout](#section-3-inference-holdout)
5. [Section 4: Data Quality & Preprocessing](#section-4-data-quality--preprocessing)
6. [Section 5: Feature Encoding & Scaling](#section-5-feature-encoding--scaling)
7. [Section 6: Final Pipeline & Export](#section-6-final-pipeline--export)

---

## Key Concepts

### What is Data Preprocessing?

Data preprocessing is the process of cleaning and transforming raw data into a format that machine learning models can understand and learn from. It's the critical first step that determines how well your model will eventually perform.

### Why is it Important?

- **Garbage in, garbage out**: If you feed a model bad data, it produces bad predictions
- **Models have assumptions**: They expect numbers, not text; no missing values; reasonable ranges
- **Real data is messy**: It contains typos, missing values, inconsistent formats, and outliers

### Data Leakage — The Silent Killer

**Data leakage** happens when information from the test set or future data "leaks" into the training process. This makes your model appear better than it actually is. Example:

- ❌ **Bad**: Include "Days for shipping (real)" as a feature — this is only known AFTER the delivery happens, but you're training a model to predict delivery time BEFORE shipping
- ✅ **Good**: Only use features available at order placement time (customer location, product type, order date)

---

## Section 1: Project Overview

### The Business Problem

E-commerce companies lose money when deliveries are late. They want to predict which orders will be late **at the moment the customer places the order** — before the order ships. This gives them time to intervene (expedite shipping, etc.).

### The Dataset

- **Source**: DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS
- **Size**: 180,520 orders (rows) × 53 attributes (columns)
- **Target variable**: `Late_delivery_risk` (0 = on-time, 1 = late)

### What This Notebook Outputs

A file called `prepared_data.pkl` containing:
- Training and test data
- A scaler (for normalizing numbers)
- An encoder (for converting text to numbers)
- The 10-row inference holdout (for final validation)

---

## Section 2: Data Loading & Initial Audit

### Step 2.1: Library Imports

```python
import pandas as pd          # Data manipulation (tables/dataframes)
import numpy as np           # Numerical arrays and math
import matplotlib.pyplot as plt  # Making charts
import seaborn as sns        # Pretty statistical charts
import pickle                # Saving/loading Python objects
import os                    # File system operations
import re                    # Regular expressions (text pattern matching)
import warnings              # Suppress warning messages
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler, OneHotEncoder  # ML tools
from sklearn.model_selection import train_test_split           # Train/test split

# Set random seed for reproducibility (same results every time)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

**Line-by-line explanation:**
- `import pandas as pd` — loads the pandas library, alias it as `pd` for convenience
- `np.random.seed(42)` — fixes the random number generator so every run produces identical results
- `warnings.filterwarnings('ignore')` — hide warning messages (not errors, just warnings)

### Step 2.2: Load the Dataset

```python
df = pd.read_csv(DATA_PATH, encoding='latin1')
```

**What's happening:**
- `pd.read_csv()` — reads a CSV file into a pandas DataFrame (think: spreadsheet)
- `encoding='latin1'` — the file uses Windows-1252 character encoding (not UTF-8), so we tell pandas how to read special characters
- `df` — the variable now holds the entire dataset

**Output:**
```
Dataset loaded: 180,519 rows × 53 columns
```

### Step 2.3: Data Type Audit

```python
print(df.dtypes.to_string())
```

**What's happening:**
- `df.dtypes` — shows the data type of each column
  - `int64` = integer (whole number)
  - `float64` = decimal number
  - `str`/`object` = text
- `to_string()` — format it nicely for printing

**Why it matters:**
- If a number is stored as text (`str`), models can't use it
- If a date is stored as text, we need to parse it
- Type mismatches often indicate data quality issues

### Step 2.4: Descriptive Statistics

```python
df.describe().T.round(2)
```

**What's happening:**
- `df.describe()` — gives statistics (min, max, mean, std dev) for each numerical column
- `.T` — transpose (flip rows and columns for readability)
- `.round(2)` — round to 2 decimal places

**What you're looking for:**
- **Min/Max**: Are there impossible values? (e.g., negative prices)
- **Mean vs Median**: Large difference suggests outliers
- **NaN count**: Missing values need handling

### Step 2.5: Missing Value Audit

```python
missing = df.isnull().mean().sort_values(ascending=False)
missing_pct = (missing * 100).round(2)
```

**What's happening:**
- `df.isnull()` — returns True where there's a missing value (NaN)
- `.mean()` — calculates the fraction of missing values per column
- `* 100` — convert to percentage
- `.sort_values(ascending=False)` — sort from most missing to least

**Key finding:**
- `Product Description`: 100% missing → **drop it entirely**
- `Order Zipcode`: 86.24% missing → **too much missing to be useful**
- `Customer Zipcode`: some missing → **impute with median**

### Step 2.6: Target Class Distribution (Chart 1)

```python
target_counts = df['Late_delivery_risk'].value_counts()
target_pct = (df['Late_delivery_risk'].value_counts(normalize=True) * 100).round(1)

print(f"  On-Time (0): {target_counts[0]:,}  ({target_pct[0]}%)")
print(f"  Late    (1): {target_counts[1]:,}  ({target_pct[1]}%)")
```

**What's happening:**
- `df['Late_delivery_risk'].value_counts()` — count how many 0s and 1s exist
- `normalize=True` — convert counts to percentages
- `{x:,}` — format number with commas (e.g., 98,977 instead of 98977)

**Key finding:**
```
On-Time (0): 81,542  (45.2%)
Late    (1): 98,977  (54.8%)
```

**Why this matters:**
- The classes are slightly imbalanced (~55% late, ~45% on-time)
- A naive model that predicts "always late" would be 55% accurate — **not good enough**
- Use F1-score and ROC-AUC instead of accuracy for evaluation

### Step 2.7: Correlation Heatmap (Chart 3)

```python
num_audit = audit_df.select_dtypes(include=[np.number])
corr = num_audit.corr()
```

**What's happening:**
- `select_dtypes(include=[np.number])` — select only numerical columns
- `.corr()` — compute Pearson correlation (how much two features move together)
- Correlation ranges from -1 to +1
  - **+1**: perfectly move together
  - **0**: no relationship
  - **-1**: move in opposite directions

**Key finding:**
```
Order Customer Id ↔ Customer Id: 1.000  (perfect duplicate!)
Sales ↔ Sales per customer: 0.990     (almost duplicate)
```

**Why it matters:**
- Highly correlated features carry the same information
- Redundant features slow down models and cause multicollinearity
- Solution: Keep only one of the correlated pair

### Step 2.8: Outlier Box Plots (Chart 4)

```python
q1 = df[col].quantile(0.25)    # 25th percentile
q3 = df[col].quantile(0.75)    # 75th percentile
iqr = q3 - q1                  # Interquartile range
n_outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
```

**What's happening:**
- **Quartiles**: divide data into 4 equal parts
- **IQR (Interquartile Range)**: spread of middle 50% of data
- **Outlier rule**: Any value beyond `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` is considered an outlier
- `|` means "OR" — count values below lower bound OR above upper bound

**Key finding:**
```
Benefit per order: 18,942 outliers (10.5%)
Product Price: 2,048 outliers (1.1%)
```

**Why it matters:**
- Outliers can distort model training (especially k-NN, neural networks)
- Solution: Winsorize — clip extreme values to reasonable bounds (1st and 99th percentile)

---

## Section 3: Inference Holdout

### Why Extract the Holdout First?

**Critical principle**: Before fitting ANY preprocessing tool (scaler, encoder), extract the holdout set. This prevents the holdout from influencing statistics.

```python
df_inference_raw = df.sample(10, random_state=RANDOM_STATE).copy()
df = df.drop(df_inference_raw.index).reset_index(drop=True)
```

**Line-by-line:**
- `df.sample(10, random_state=42)` — randomly select 10 rows (reproducibly)
- `.copy()` — make a copy (don't reference the original)
- `df.drop(df_inference_raw.index)` — remove those 10 rows from the working dataset
- `.reset_index(drop=True)` — renumber the remaining rows (0, 1, 2, ... instead of skipping)

**Result:**
```
Inference holdout: 10 rows extracted
Working dataset after holdout removal: 180,509 rows
```

**Why it matters:**
- The 10 rows are completely isolated
- No statistic (scaler median, encoder vocabulary) is computed from them
- This matches real deployment: new orders are processed by a frozen pipeline

---

## Section 4: Data Quality & Preprocessing

### 4.1: Remove Post-Shipment Leakage

**CRITICAL DECISION**: The model must predict using only information available **at order placement**.

```python
LEAKAGE_COLS = [
    'Days for shipping (real)',    # actual transit days — only known AFTER delivery
    'Delivery Status',              # 'Late delivery', 'Advance shipping' — post-delivery
    'shipping date (DateOrders)',   # when item actually left warehouse — AFTER order
    'Order Status',                 # 'COMPLETE', 'CLOSED' — post-fulfillment
]

df = df.drop(columns=leakage_dropped)
```

**Why each column leaks:**
1. **Days for shipping (real)**: Example: You see an order will take 5 days to ship. If you use this to predict "late", you've cheated — you're seeing the future!
2. **Delivery Status**: Direct outcome — violates the principle of predicting from features
3. **shipping date**: Only recorded when the warehouse ships it
4. **Order Status**: Only changes after the order is complete

**After removal:**
```
Columns: 53 → 49
```

### 4.2: Remove Identifiers & PII

**Principle**: ID columns and personal information carry no predictive signal.

```python
# Rule 1: Drop columns that are >95% unique
high_unique = [
    c for c in df.columns
    if df[c].nunique() / n_rows > 0.95
]

# Rule 2: Drop columns with keyword names
ID_KEYWORDS = ['id', 'email', 'password', 'image', 'url', 'fname', 'lname', 'phone']
name_based = [
    c for c in df.columns
    if any(kw in c.lower() for kw in ID_KEYWORDS)
]

# Rule 3: Drop columns with email/URL patterns in content
EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
URL_RE = re.compile(r'https?://|www\.')
```

**Line-by-line:**
- `df[c].nunique()` — count unique values in column `c`
- `nunique() / n_rows > 0.95` — if 95%+ are unique, it's an ID column
- `any(kw in c.lower() for kw in ID_KEYWORDS)` — check if column name contains any keyword (converted to lowercase)
- `re.compile()` — create a regex pattern
- `contains(EMAIL_RE.pattern, regex=True)` — check if column values match email pattern

**Dropped columns:**
```
14 identifier columns removed:
  Order Id, Customer Id, Customer Email, Product Card Id, etc.
```

**Result:**
```
Columns remaining: 35 (down from 49)
```

### 4.3: Parse Dates & Extract Temporal Features

```python
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df['order_month']      = df[date_col].dt.month.astype('Int64')
df['order_dayofweek']  = df[date_col].dt.dayofweek.astype('Int64')
df['order_quarter']    = df[date_col].dt.quarter.astype('Int64')
df = df.drop(columns=[date_col])
```

**Line-by-line:**
- `pd.to_datetime()` — parse text date into datetime object
- `errors='coerce'` — invalid dates become NaN (not an error)
- `dt.month` — extract month (1-12) from datetime
- `dt.dayofweek` — extract day of week (0=Mon, 6=Sun)
- `dt.quarter` — extract quarter (1-4)
- `astype('Int64')` — convert to integer (nullable)
- `.drop(columns=[date_col])` — remove original date column (no longer needed)

**Why extract temporal features:**
- **Month**: Holiday season (Nov-Dec) might have more delays
- **Day of week**: Weekend orders might process slower
- **Quarter**: Q4 is busy season, more delays possible
- Raw timestamp as text is useless to models

**Result:**
```
'order date (DateOrders)' parsed → 3 temporal features extracted
Dataset shape: (180509, 37)  [was 35, added 3, minus 1 old date = 37]
```

### 4.4: Remove Exact Duplicates

```python
n_dupes = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
```

**What's happening:**
- `df.duplicated()` — return True for rows that are exact copies of earlier rows
- `.sum()` — count how many duplicates exist
- `drop_duplicates()` — remove them
- `reset_index(drop=True)` — renumber rows after deletion

**Result:**
```
Exact duplicate rows detected: 0
```

(None in this dataset, but good to check!)

### 4.5: Impute Missing Values

**Three-tier strategy:**

```python
# Step 1: Drop columns with >40% missing
drop_high_missing = missing_frac[missing_frac > 0.40].index.tolist()
df = df.drop(columns=drop_high_missing)

# Step 2 & 3: Impute remaining missing values
for col in cols_with_missing:
    if df[col].dtype.kind in ('i', 'u', 'f', 'c'):
        # Numerical: use median
        val = df[col].median()
        df[col] = df[col].fillna(val)
        imputation_values[col] = float(val)
    else:
        # Categorical: use mode (most common)
        val = df[col].mode()[0]
        df[col] = df[col].fillna(val)
        imputation_values[col] = str(val)
```

**Line-by-line:**
- `missing_frac[missing_frac > 0.40]` — filter columns with >40% missing
- `.dtype.kind in ('i', 'u', 'f', 'c')` — check if numeric (i=signed int, f=float, etc.)
- `df[col].median()` — middle value (robust to outliers)
- `df[col].fillna(val)` — replace NaN with computed value
- `df[col].mode()[0]` — most frequently occurring category

**Why median, not mean?**
- Mean is distorted by extreme outliers
- Median is the middle value — unaffected by extremes

**Why store `imputation_values`?**
- Inference holdout must use same values
- Don't recompute from inference data (leakage!)

**Result:**
```
Dropped (>40% missing): ['Order Zipcode', 'Product Description']
[Median] 'Customer Zipcode' ← 19380.0000
Dataset shape after imputation: (180509, 35)
Remaining missing values: 0
```

### 4.6: Winsorize Outliers

```python
for col in num_cols_for_winsor:
    p01 = df[col].quantile(0.01)   # 1st percentile
    p99 = df[col].quantile(0.99)   # 99th percentile
    df[col] = df[col].clip(lower=p01, upper=p99)
```

**What's happening:**
- `quantile(0.01)` — value where 1% of data is below it
- `quantile(0.99)` — value where 99% of data is below it
- `clip(lower=p01, upper=p99)` — cap values at these bounds
  - Values below p01 → set to p01
  - Values above p99 → set to p99
  - Values between → unchanged

**Example:**
```
Benefit per order: p01=-415.616, p99=184.230, clipped 3,605 values
```

(Any value < -415.616 or > 184.230 gets capped)

**Why winsorize, not delete?**
- **Preserves dataset size**: Removing outliers loses data
- **Real events**: Negative profit means loss-making order (real, not error)
- **Prevents distortion**: Outliers dominate distance-based models (k-NN)

### 4.7: Feature Selection — Remove Near-Zero Variance & High Correlation

```python
# Filter A: Near-zero variance (>99% same value)
low_var_cols = [
    c for c in num_cols_check
    if df[c].value_counts(normalize=True).iloc[0] > 0.99
]
df = df.drop(columns=low_var_cols)

# Filter B: High correlation (>0.95)
corr_matrix = df[num_cols_corr].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

for col in upper_tri.columns:
    correlated = upper_tri.index[upper_tri[col] > 0.95].tolist()
    for c2 in correlated:
        if c2 not in high_corr_drop:
            high_corr_drop.append(c2)

df = df.drop(columns=high_corr_drop)
```

**Line-by-line:**
- `value_counts(normalize=True)` — fraction of each unique value
- `.iloc[0]` — most common value's fraction
- `> 0.99` — if one value accounts for >99%, it's near-constant
- `corr().abs()` — absolute correlation (ignore sign, just magnitude)
- `np.triu(..., k=1)` — upper triangle of correlation matrix (avoid double-counting)
- `[correlated]` — columns highly correlated with current column
- Drop the **second** occurrence (keep first)

**Results:**
```
Filter A — Dropped: ['Product Status'] (always 0)
Filter B — Dropped 6 columns:
  'Customer Zipcode' (corr=0.961 with Longitude)
  'Sales per customer' (corr=0.988 with Sales)
  'Sales' (corr=0.988 with Order Item Total)
  'Benefit per order' (corr=1.000 with Order Profit Per Order)
  'Order Item Product Price' (corr=1.000 with Product Price)
  'order_month' (corr=0.971 with order_quarter)

Dataset shape: (180509, 28)
```

**Why remove these:**
- Redundant features add noise without benefit
- Slow down distance-based models
- Cause multicollinearity in linear models

---

## Section 5: Feature Encoding & Scaling

### 5.1: Separate Categorical & Numerical Columns

```python
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
```

**What's happening:**
- `select_dtypes(include='object')` — pick columns with text (object = string)
- `select_dtypes(include=[np.number])` — pick columns with numbers
- `.columns.tolist()` — convert to a list of column names

**Result:**
```
Categorical columns (15): Type, Category Name, Customer City, ...
Numerical columns (12): Days for shipment, Latitude, Longitude, ...
```

### 5.2: Group Rare Categories

```python
RARE_THRESHOLD = 0.005  # 0.5%

for col in cat_cols:
    freq = X[col].value_counts(normalize=True)
    rare_cats = freq[freq < RARE_THRESHOLD].index.tolist()
    if rare_cats:
        X[col] = X[col].replace(rare_cats, 'Other')
```

**Line-by-line:**
- `value_counts(normalize=True)` — fraction of each category
- `freq < RARE_THRESHOLD` — categories appearing <0.5% of time
- `.replace(rare_cats, 'Other')` — replace rare categories with "Other"

**Example:**
```
'Customer City': 547 rare cities merged → 17 unique cities remain
'Product Name': 109 rare products merged → 10 unique products remain
```

**Why group rare categories:**
- Rare categories → one-hot encoding creates mostly-zero columns
- Zero-variance features add noise
- Grouping keeps feature space manageable

### 5.3: One-Hot Encoding Concept

**What is One-Hot Encoding (OHE)?**

Problem: Categories like "Type" has values [DEBIT, TRANSFER, CASH, CHECK]. Models need numbers, not text.

Solution: Convert each category into binary columns:

**Before:**
```
Type
DEBIT
TRANSFER
CASH
DEBIT
TRANSFER
```

**After OHE:**
```
Type_DEBIT  Type_TRANSFER  Type_CASH  Type_CHECK
    1            0            0          0
    0            1            0          0
    0            0            1          0
    1            0            0          0
    0            1            0          0
```

Each category becomes a binary column (1 = that category, 0 = not that category).

**Why not ordinal encoding (0, 1, 2, 3)?**
- Implies ordering (0 < 1 < 2 < 3) which is false for nominal categories
- Models would think CASH > DEBIT, which is meaningless
- OHE treats all categories equally

### 5.4: What is Scaling?

**Problem**: Features have different ranges:
- Latitude: -33 to +48 (range ≈ 80)
- Discount Rate: 0 to 0.25 (range ≈ 0.25)

k-NN uses Euclidean distance. Without scaling, Latitude dominates because it's bigger numbers.

**Solution: RobustScaler**

```python
scaler = RobustScaler()
scaler.fit(X_train_raw[num_cols])
X_train_num = scaler.transform(X_train_raw[num_cols])
X_test_num = scaler.transform(X_test_raw[num_cols])
```

**How RobustScaler works:**
```
scaled_value = (original_value - median) / IQR
```

- **Median**: middle value (resistant to outliers)
- **IQR**: interquartile range (spread of middle 50%)

**Example:**
```
Benefit per order: median = 31.52, IQR = 57.8
Original: 100 → Scaled: (100 - 31.52) / 57.8 ≈ 1.18
Original: 10  → Scaled: (10 - 31.52) / 57.8 ≈ -0.37
```

All scaled values cluster around 0, comparably scaled.

**Why RobustScaler, not StandardScaler?**
- **StandardScaler**: uses mean/std dev (distorted by outliers)
- **RobustScaler**: uses median/IQR (ignores outliers)

### 5.5: Fit on Train, Apply to Test (NO LEAKAGE!)

```python
# Fit encoder on TRAINING data only
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X_train_raw[cat_cols])

# Transform both splits
X_train_cat = encoder.transform(X_train_raw[cat_cols])
X_test_cat = encoder.transform(X_test_raw[cat_cols])
```

**Critical principle:**
1. **Fit** encoder on training data only
   - Learn category vocabulary from training set
   - Compute statistics (median/IQR) from training set only
2. **Transform** test data using the fitted encoder
   - Use frozen vocabulary
   - Use frozen statistics

**What if test set has a category not in training?**
- `handle_unknown='ignore'` → set all indicators to 0
- No error, graceful degradation

**Why this matters:**
- Test set statistics don't influence scaling
- Matches real deployment: new data processed by frozen pipeline
- Prevents information leakage

---

## Section 6: Train-Test Split, Encoding, Scaling & Artifact Export

### 6.1: Stratified 80/20 Split

```python
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
```

**What's happening:**
- `test_size=0.2` — 20% for testing, 80% for training
- `stratify=y` — preserve class distribution in both sets
- `random_state=42` — reproducible split

**Why stratify?**
- Dataset has ~55% late, ~45% on-time
- Random split might give test set 60% late (or 50%)
- Stratified ensures both have ~55% late

**Result:**
```
Training set: 144,407 rows (54.83% late)
Test set: 36,102 rows (54.83% late)
```

Perfect preservation of class balance!

### 6.2: Fit & Transform Encoder

```python
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X_train_raw[cat_cols])

X_train_cat = encoder.transform(X_train_raw[cat_cols])
X_test_cat = encoder.transform(X_test_raw[cat_cols])
cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()
```

**Line-by-line:**
- `encoder.fit(...)` — learn category vocabulary from training data
- `encoder.transform(...)` — convert categories to OHE columns
- `get_feature_names_out()` — get names of generated columns (e.g., "Type_DEBIT", "Type_TRANSFER")
- `.tolist()` — convert to Python list

**Result:**
```
Categorical columns encoded: 15 → 220 OHE columns
X_train_cat shape: (144407, 220)
X_test_cat  shape: (36102, 220)
```

### 6.3: Fit & Transform Scaler

```python
scaler = RobustScaler()
scaler.fit(X_train_raw[num_cols])

X_train_num = scaler.transform(X_train_raw[num_cols])
X_test_num = scaler.transform(X_test_raw[num_cols])

X_train_final = np.concatenate([X_train_num, X_train_cat], axis=1)
X_test_final = np.concatenate([X_test_num, X_test_cat], axis=1)
```

**What's happening:**
- `scaler.fit(...)` — compute median/IQR from training numerical data
- `scaler.transform(...)` — apply scaling to both train and test
- `np.concatenate(..., axis=1)` — join arrays horizontally (side-by-side)
  - `axis=1` means join columns (not rows)
  - Result: [scaled_numbers | OHE_columns]

**Result:**
```
X_train final: (144407, 232)  [12 numerical + 220 OHE]
X_test final: (36102, 232)    [same structure]
```

### 6.4: Apply Pipeline to Inference Holdout

```python
# Apply all transformations using FITTED encoder and scaler
inf_cat = encoder.transform(inf_X[inf_cat_cols].reindex(columns=cat_cols, fill_value='Other'))
inf_num = scaler.transform(inf_X[inf_num_cols].reindex(columns=num_cols, fill_value=0))

df_inference_processed = np.concatenate([inf_num, inf_cat], axis=1)
```

**Key points:**
- Use `.reindex(fill_value='Other')` — if a column is missing, fill with 'Other'
- Use encoder and scaler WITHOUT refitting
- Result has same 232 features as training data

**Result:**
```
Inference holdout processed: (10, 232)
Feature vector length matches training: True
True labels: [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
```

### 6.5: Save Everything to Pickle

```python
artifact = {
    'X_train': X_train_final,
    'X_test': X_test_final,
    'y_train': y_train.reset_index(drop=True),
    'y_test': y_test.reset_index(drop=True),
    'df_inference_raw': df_inference_raw,
    'df_inference_processed': df_inference_processed,
    'scaler': scaler,
    'encoder': encoder,
    'feature_names': feature_names,
    'target_name': TARGET,
    '_meta': { ... metadata ... }
}

with open(ARTIFACT_PATH, 'wb') as f:
    pickle.dump(artifact, f)
```

**What's a pickle file?**
- Binary format for saving Python objects
- Includes everything: arrays, sklearn objects, lists
- Models notebook can load it all: `artifact = pickle.load(f)`

**What's saved:**
- **X_train/X_test**: Feature matrices (numpy arrays)
- **y_train/y_test**: Target labels (0/1)
- **df_inference_raw**: Original 10 holdout rows (for display)
- **df_inference_processed**: Processed 10 rows (for prediction)
- **scaler**: Fitted RobustScaler (call `.transform()` only)
- **encoder**: Fitted OneHotEncoder (call `.transform()` only)
- **feature_names**: List of 232 column names
- **_meta**: Metadata (number of features, column names, etc.)

### 6.6: Final Sanity Checks

```python
# Check 1: No NaN values in final arrays
assert not np.isnan(X_train_final).any(), "NaN detected in X_train!"
assert not np.isnan(X_test_final).any(), "NaN detected in X_test!"

# Check 2: Shapes match
assert X_train_final.shape[1] == X_test_final.shape[1] == len(feature_names)

# Check 3: Inference holdout shape matches
assert df_inference_processed.shape[1] == X_train_final.shape[1]

# Check 4: Target is binary
assert set(y_train.unique()) <= {0, 1}, "Non-binary values in y_train!"

# Check 5: Class balance preserved
train_late_pct = y_train.mean() * 100
test_late_pct = y_test.mean() * 100
assert abs(train_late_pct - test_late_pct) < 1.0, "Stratification failed!"

# Check 6: Pickle loads
with open(ARTIFACT_PATH, 'rb') as f:
    loaded = pickle.load(f)
assert set(loaded.keys()) >= {'X_train','X_test','y_train','y_test', ...}
```

**All checks pass:**
```
✓ No NaN values in X_train or X_test
✓ Feature count consistent: 232 columns
✓ Inference holdout feature count matches: 232
✓ Target is binary: values = [0, 1]
✓ Class balance preserved: train 54.8% late | test 54.8% late
✓ Pickle loads successfully with all required keys
```

---

## Complete Data Flow Diagram

```
Raw CSV (180,520 rows)
    ↓
[Extract 10-row inference holdout]
    ↓ (180,510 working rows)
[Remove leakage columns: Days for shipping, Delivery Status, etc.]
    ↓ (49 columns)
[Remove identifiers: IDs, emails, passwords, images]
    ↓ (35 columns)
[Parse dates → extract month, dayofweek, quarter]
    ↓ (37 columns)
[Remove exact duplicates]
    ↓ (180,509 rows — no duplicates found)
[Impute missing values: median (numeric), mode (categorical)]
    ↓ (0 missing values)
[Winsorize outliers: clip at 1st/99th percentiles]
    ↓
[Remove near-zero variance features]
    ↓
[Remove high-correlation features (>0.95)]
    ↓ (28 columns)
[Group rare categories (<0.5%) → 'Other']
    ↓ (27 features, 27 target)
[STRATIFIED 80/20 SPLIT]
    ↓
    Train (144,407 rows)          Test (36,102 rows)
        ↓                             ↓
    [Fit encoder]             [Transform with fitted encoder]
    [Fit scaler]              [Transform with fitted scaler]
        ↓                             ↓
    X_train: (144407, 232)     X_test: (36102, 232)
    
[Apply frozen encoder & scaler to inference holdout]
    ↓
    Inference: (10, 232)
    
[Save to artifact/prepared_data.pkl]
    ↓ (ready for model training)
```

---

## Key Takeaways

| Concept | Explanation |
|---------|-------------|
| **Data Leakage** | Using information that wouldn't be available in production (e.g., future shipping times) |
| **Stratified Split** | Preserve class distribution in train and test sets (~55% late in both) |
| **Fit vs Transform** | Fit encoder/scaler on TRAINING only, then transform TEST (no leakage) |
| **Winsorization** | Clip extreme outliers to reasonable bounds (e.g., 1st/99th percentile) |
| **One-Hot Encoding** | Convert categories (text) to binary columns (numbers) |
| **RobustScaler** | Scale using median/IQR instead of mean/std (resistant to outliers) |
| **Inference Holdout** | 10 rows held completely separate to test final model on unseen data |
| **Pickle Artifact** | Binary file containing all preprocessed data and fitted objects |

---

## What Happens Next

The `prepared_data.pkl` artifact is now ready for:
1. **Model Training**: Train k-NN, Perceptron, LinearSVC, Random Forest, XGBoost
2. **Model Evaluation**: Test on holdout set, compute F1-score, ROC-AUC, etc.
3. **Inference**: Apply trained models to the 10 holdout rows to get predictions

All downstream notebooks load this artifact—no preprocessing is repeated.
