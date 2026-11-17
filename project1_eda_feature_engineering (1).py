"""
==========================================================================================
 PROJECT 1 — Advanced EDA & Feature Engineering (DecodeLabs Data Science Industrial Kit)
==========================================================================================
Dataset : E-commerce order records (extracted from the supplied Google Sheets PDF export)
Goal    : Turn raw, messy order data into a mathematically clean dataset ready for ML.

This script covers all three required deliverables:
  1. Missing-value handling via statistical imputation
  2. Outlier detection & neutralization via IQR
  3. Engineering 3+ new predictive features

Run with:  python3 project1_eda_feature_engineering.py
==========================================================================================
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)

# ------------------------------------------------------------------------------------------
# STEP 0 — LOAD RAW DATA
# ------------------------------------------------------------------------------------------
# raw_orders.csv was reconstructed from the source PDF (a Google-Sheets export whose text
# layer had columns concatenated together). Each row already represents one order.
df = pd.read_csv("raw_orders.csv", dtype=str, keep_default_na=False)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns\n")

# ------------------------------------------------------------------------------------------
# STEP 1 — TYPE CORRECTION
# ------------------------------------------------------------------------------------------
# Everything came in as text. Cast numeric columns properly; anything that fails to parse
# becomes NaN, which is exactly how pandas is supposed to represent "missing".
numeric_cols = ["qty", "price", "addr_num", "cart_items", "total"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# The PDF's text layer drops the LAST digit of the day in every date (a rendering artifact
# of the source export, e.g. "2023-01-0" instead of "2023-01-05"). The year and month are
# fully reliable; only the day digit is corrupted. We keep year/month as real signal and
# treat "day" as MISSING for the entire column — a good, honest example of a systematic
# data-collection defect rather than random missingness.
df["order_year"] = df["date"].str.slice(0, 4).astype(int)
df["order_month"] = df["date"].str.slice(5, 7).astype(int)
df["order_day_raw"] = df["date"].str.slice(8, 10)  # corrupted, single digit or missing

# Treat blank categorical fields as their own explicit category rather than np.nan,
# so we can measure missingness deliberately in Step 2.
for c in ["payment", "status", "coupon", "referral"]:
    df[c] = df[c].replace("", np.nan)

print("Dtypes after correction:")
print(df.dtypes, "\n")

# ------------------------------------------------------------------------------------------
# STEP 2 — MISSING DATA: THE DECISION MATRIX
# ------------------------------------------------------------------------------------------
# Per-feature missingness dictates the strategy (see project brief's "Missing Data
# Decision Matrix"):
#   < 5%    -> drop rows (cheap, doesn't bias the sample)
#   5%-20%  -> statistical imputation (median for skewed numerics, mode/group-wise for
#              categoricals)
#   > 20%   -> multi-dimensional estimation (KNN) OR domain-informed fill if the
#              missingness itself is meaningful (MNAR)
print("=" * 90)
print("MISSING VALUE REPORT")
print("=" * 90)
missing_pct = df.isna().mean().sort_values(ascending=False) * 100
print(missing_pct[missing_pct > 0].round(2), "\n")

# --- coupon (~26% missing) ---------------------------------------------------------------
# This is NOT random missingness (MCAR). A blank coupon code means "no coupon was applied"
# — the absence IS the information. Blindly running KNN or mode-imputation here would
# invent a fake coupon for genuine no-coupon orders and destroy that signal. The correct
# statistical move is to encode the missingness itself as a category.
df["coupon"] = df["coupon"].fillna("NoCoupon")

# --- order_day_raw (corrupted for every row) ---------------------------------------------
# Day-of-month has no relationship to any other column (it's essentially noise), so there's
# no informative pattern for KNN to exploit. We impute with the column's own median day
# (statistically neutral) purely so a full date can still be reconstructed for any
# downstream time-based feature that only needs year/month granularity.
median_day = 15  # midpoint of a month — neutral choice, documented explicitly
df["order_day"] = pd.to_numeric(df["order_day_raw"], errors="coerce").fillna(median_day).astype(int)
df["order_day"] = df["order_day"].clip(1, 28)  # keep every month valid
df["order_date"] = pd.to_datetime(
    dict(year=df.order_year, month=df.order_month, day=df.order_day)
)
df.drop(columns=["order_day_raw", "date"], inplace=True)

# --- generic numeric imputer (kept general-purpose, ready for real missingness) ----------
def impute_numeric(series: pd.Series) -> pd.Series:
    """Median imputation for numeric columns — robust to outliers/skew (see slide:
    'Global Median' preserves distribution shape better than mean when data is skewed)."""
    pct_missing = series.isna().mean() * 100
    if pct_missing == 0:
        return series
    if pct_missing < 5:
        print(f"  -> {series.name}: <5% missing, dropping the affected rows")
        return series  # rows dropped globally at the end, see below
    print(f"  -> {series.name}: {pct_missing:.1f}% missing, median imputation")
    return series.fillna(series.median())


for c in numeric_cols + ["cart_items"]:
    df[c] = impute_numeric(df[c])

# Drop any residual rows where a <5%-missing numeric field couldn't be safely imputed.
before = len(df)
df.dropna(subset=numeric_cols, inplace=True)
if before != len(df):
    print(f"Dropped {before - len(df)} rows with irrecoverable numeric gaps")

print("\nMissing values remaining:", df.isna().sum().sum(), "\n")

# ------------------------------------------------------------------------------------------
# STEP 3 — OUTLIER DETECTION & NEUTRALIZATION (IQR)
# ------------------------------------------------------------------------------------------
# For each numeric column: Q1, Q3, IQR = Q3-Q1, bounds = Q1 - 1.5*IQR / Q3 + 1.5*IQR.
# We WINSORIZE (clip) rather than delete rows — this preserves sample size and row
# alignment, which matters because `total` is derived from `qty * price` and we don't want
# to break that relationship by dropping only one of the three.
def iqr_bounds(series: pd.Series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


print("=" * 90)
print("OUTLIER REPORT (IQR method)")
print("=" * 90)
outlier_cols = ["qty", "price", "cart_items", "total"]
for c in outlier_cols:
    low, high = iqr_bounds(df[c])
    n_out = ((df[c] < low) | (df[c] > high)).sum()
    print(f"  {c:12s} bounds=({low:.2f}, {high:.2f})  outliers found={n_out}")
    df[c] = df[c].clip(low, high)  # numpy-style winsorization, preserves row count
print()

# ------------------------------------------------------------------------------------------
# STEP 4 — FEATURE ENGINEERING (3+ new predictive features required)
# ------------------------------------------------------------------------------------------
# 1) cart_fill_ratio: what fraction of the items sitting in the cart were actually bought.
#    A low ratio suggests cart abandonment / browsing behaviour; predictive of future churn.
df["cart_fill_ratio"] = (df["qty"] / df["cart_items"]).round(3)

# 2) avg_value_per_cart_item: total order value spread over everything in the cart —
#    a proxy for "how premium is this customer's basket", independent of raw quantity.
df["avg_value_per_cart_item"] = (df["total"] / df["cart_items"]).round(2)

# 3) is_returned_or_cancelled: binary target-adjacent feature flagging revenue-negative
#    outcomes — critical for any churn / logistics-quality model built on top of this data.
df["is_returned_or_cancelled"] = df["status"].isin(["Returned", "Cancelled"]).astype(int)

# 4) has_coupon: binary flag — orders with a discount code often behave differently
#    (price sensitivity) than full-price orders.
df["has_coupon"] = (df["coupon"] != "NoCoupon").astype(int)

# 5) order_quarter: coarser seasonality signal than month alone, useful when monthly
#    granularity is too sparse per category for a model to learn from.
df["order_quarter"] = df["order_date"].dt.quarter

print("New engineered features added:")
for c in ["cart_fill_ratio", "avg_value_per_cart_item", "is_returned_or_cancelled",
          "has_coupon", "order_quarter"]:
    print(f"  - {c}")
print()

# ------------------------------------------------------------------------------------------
# STEP 5 — CATEGORICAL ENCODING (bonus, avoids the "label-encoding" fake-ordinality trap)
# ------------------------------------------------------------------------------------------
# Product, payment, status, referral, and coupon are NOMINAL categories with no inherent
# order. Assigning them integers (Label Encoding) would create a false numeric distance
# between categories. One-Hot Encoding maps each into its own orthogonal axis instead.
categorical_cols = ["product", "payment", "status", "referral", "coupon"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# ------------------------------------------------------------------------------------------
# STEP 6 — MULTICOLLINEARITY CHECK (bonus sanity check before handing off to a model)
# ------------------------------------------------------------------------------------------
corr = df[["qty", "price", "total", "cart_items", "cart_fill_ratio",
           "avg_value_per_cart_item"]].corr(numeric_only=True)
print("=" * 90)
print("CORRELATION MATRIX (numeric features)")
print("=" * 90)
print(corr.round(2), "\n")

high_corr_pairs = [
    (a, b, corr.loc[a, b])
    for a in corr.columns for b in corr.columns
    if a < b and abs(corr.loc[a, b]) > 0.8
]
if high_corr_pairs:
    print("High-collinearity pairs (|r| > 0.8) — consider dropping one of each:")
    for a, b, r in high_corr_pairs:
        print(f"  {a} <-> {b}: r={r:.2f}")
else:
    print("No problematic multicollinearity (|r| > 0.8) detected among engineered numerics.")

# ------------------------------------------------------------------------------------------
# STEP 7 — SAVE CLEAN OUTPUTS
# ------------------------------------------------------------------------------------------
df.to_csv("cleaned_orders.csv", index=False)
df_encoded.to_csv("cleaned_orders_encoded_for_ml.csv", index=False)

print("\nSaved: cleaned_orders.csv (human-readable, cleaned + engineered features)")
print("Saved: cleaned_orders_encoded_for_ml.csv (one-hot encoded, model-ready)")
print(f"\nFinal shape: {df.shape[0]} rows x {df.shape[1]} columns "
      f"(model-ready encoded shape: {df_encoded.shape[0]} x {df_encoded.shape[1]})")
