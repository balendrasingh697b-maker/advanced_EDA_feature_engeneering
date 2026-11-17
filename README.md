# Project 1: Advanced EDA & Feature Engineering
### DecodeLabs – Data Science Industrial Kit

## 📌 Project Overview

This project focuses on transforming raw, messy e-commerce order records into a clean, structured, and machine-learning-ready dataset.

The dataset was reconstructed from a supplied Google Sheets PDF export. Since the original PDF contained formatting and text-extraction issues, the data required preprocessing, type correction, missing-value treatment, outlier handling, and feature engineering.

The primary objective is to demonstrate practical data preprocessing techniques used in real-world Data Science and Machine Learning workflows.

## 🎯 Project Objectives

The project addresses the following key objectives:

1. **Missing Value Handling:** Identify missing values and apply appropriate statistical or domain-based imputation techniques.
2. **Outlier Detection & Treatment:** Detect extreme values using the Interquartile Range (IQR) method and neutralize them through Winsorization.
3. **Feature Engineering:** Create meaningful predictive features from existing columns.
4. **Categorical Encoding:** Convert categorical variables into numerical representations using One-Hot Encoding.
5. **Correlation Analysis:** Identify highly correlated numerical variables before model development.
6. **Data Export:** Generate clean and encoded datasets ready for further machine learning applications.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Jupyter Notebook / Python | Development and execution |

## 📂 Project Structure

```text
Project-1/
│
├── raw_orders.csv
│
├── project1_eda_feature_engineering.py
│
├── cleaned_orders.csv
│
├── cleaned_orders_encoded_for_ml.csv
│
└── README.md
```

## ⚙️ Project Workflow

### Step 1: Data Loading & Type Correction

- Loaded the raw dataset using Pandas.
- Initially imported all columns as strings to preserve the original values.
- Converted numerical columns into appropriate numeric data types using `pd.to_numeric()`.
- Handled invalid numerical entries by converting them into missing values.
- Extracted year and month information from corrupted date strings.

### Step 2: Missing Value Analysis & Handling

Calculated the percentage of missing values in each column.

The following decision strategy was considered:

| Missing Percentage | Proposed Strategy |
|---|---|
| Less than 5% | Drop affected rows where appropriate |
| 5%–20% | Statistical imputation |
| More than 20% | KNN or domain-informed treatment |

**Imputation techniques implemented:**

- **Median Imputation:** Used for numerical columns because the median is less sensitive to extreme values than the mean.
- **Domain-Based Imputation:** Missing coupon values were replaced with `NoCoupon`, since an empty coupon field represents an order without a discount.
- **Date Handling:** Since the day component was corrupted throughout the dataset, a documented neutral value of 15 was used, with day values constrained to 1–28.

### Step 3: Outlier Detection & Treatment

Implemented the Interquartile Range (IQR) method to identify extreme numerical observations.

**Mathematical formula:**

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

The following columns were examined:

- `qty`
- `price`
- `cart_items`
- `total`

**Treatment: Winsorization**

Instead of deleting outlier records, values outside the calculated boundaries were clipped to the respective lower and upper bounds.

This approach helps preserve the dataset's row count while reducing the influence of extreme observations.

### Step 4: Feature Engineering

Five new features were created to extract additional information from the existing dataset.

| Feature | Description |
|---|---|
| `cart_fill_ratio` | Ratio of purchased quantity to total cart items |
| `avg_value_per_cart_item` | Average order value per cart item |
| `is_returned_or_cancelled` | Binary indicator for returned or cancelled orders |
| `has_coupon` | Indicates whether a coupon was applied |
| `order_quarter` | Extracts the quarter of the order date |

**Feature calculations:**

1. Cart Fill Ratio

```python
df["cart_fill_ratio"] = df["qty"] / df["cart_items"]
```

2. Average Value Per Cart Item

```python
df["avg_value_per_cart_item"] = df["total"] / df["cart_items"]
```

3. Returned or Cancelled Indicator

```python
df["is_returned_or_cancelled"] = (
    df["status"].isin(["Returned", "Cancelled"]).astype(int)
)
```

4. Coupon Indicator

```python
df["has_coupon"] = (df["coupon"] != "NoCoupon").astype(int)
```

5. Order Quarter

```python
df["order_quarter"] = df["order_date"].dt.quarter
```

### Step 5: Categorical Encoding

Applied One-Hot Encoding using Pandas `get_dummies()`.

Categorical columns encoded:

- Product
- Payment
- Status
- Referral
- Coupon

One-Hot Encoding was selected because these variables represent nominal categories without a natural numerical order.

```python
df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=False
)
```

### Step 6: Multicollinearity & Correlation Analysis

Generated a correlation matrix for selected numerical features.

The following variables were examined:

- Quantity
- Price
- Total
- Cart Items
- Cart Fill Ratio
- Average Value Per Cart Item

Pairs with an absolute correlation coefficient greater than 0.8 were flagged for further investigation.

This helps identify potentially redundant features before model training.

### Step 7: Exporting Cleaned Data

Two output datasets are generated.

| Output File | Description |
|---|---|
| `cleaned_orders.csv` | Cleaned dataset with engineered features and readable categorical columns |
| `cleaned_orders_encoded_for_ml.csv` | One-Hot Encoded dataset for downstream ML workflows |

## 🚀 Installation & Execution

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd Project-1
```

### 3. Install Dependencies

```bash
pip install pandas numpy
```

### 4. Place the Dataset

Ensure that `raw_orders.csv` is available in the same directory as the Python script.

### 5. Run the Project

```bash
python project1_eda_feature_engineering.py
```

After successful execution, the following files will be generated:

- `cleaned_orders.csv`
- `cleaned_orders_encoded_for_ml.csv`

## 📊 Key Concepts Demonstrated

- Exploratory Data Analysis (EDA)
- Data Cleaning and Preprocessing
- Missing Value Analysis
- Median Imputation
- Domain-Based Imputation
- Outlier Detection using IQR
- Winsorization
- Feature Engineering
- Date-Time Feature Extraction
- One-Hot Encoding
- Correlation Analysis
- Multicollinearity Detection
- Machine Learning Data Preparation

## 💡 Key Learning Outcomes

Through this project, I gained practical experience in:

- Working with real-world, imperfect datasets.
- Selecting missing-value treatment strategies based on data characteristics.
- Detecting and handling extreme numerical observations.
- Creating meaningful features from raw business data.
- Preparing categorical and numerical variables for machine learning.
- Understanding the importance of data quality before model development.

## 🔮 Future Scope

The cleaned dataset can be extended into several machine learning applications:

- Customer Purchase Prediction
- Customer Churn Prediction
- Order Cancellation Prediction
- Product Demand Forecasting
- Customer Segmentation
- Sales and Revenue Prediction

**Note:** These are potential future applications, not models trained as part of the current project.

---

## 👨‍💻 Author

**Balendra Singh**

MCA Student | Data Science & Artificial Intelligence Enthusiast

**Project:** Advanced EDA & Feature Engineering  
**Program:** DecodeLabs – Data Science Industrial Kit
