"""
==============================================================================
 SIMPLE LINEAR REGRESSION PROJECT: Sales vs. Newspaper Advertising
==============================================================================
Goal: Predict Sales based on Newspaper Advertising using a single-variable
      (simple) linear regression model.

Works out of the box with the classic "Advertising.csv" dataset
(columns: Newspaper, Sales). If your CSV has different column
names, just update the CONFIG section below — nothing else needs to change.
==============================================================================
"""

# -----------------------------------------------------------------------
# 0. CONFIGURATION — change these two lines to match your CSV
# -----------------------------------------------------------------------
CSV_PATH = "C:\\Users\\Ishita Bhingare\\Downloads\\Advertising.csv"     # path to your downloaded dataset
X_COLUMN = "Newspaper"     # the predictor (independent variable)
Y_COLUMN = "Sales"              # the target (dependent variable)


# -----------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cosmetic settings for nicer plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


# -----------------------------------------------------------------------
# 2. DATA PREPARATION — load and clean the dataset
# -----------------------------------------------------------------------
print("=" * 60)
print("STEP 1: DATA PREPARATION")
print("=" * 60)

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("C:\\Users\\Ishita Bhingare\\Downloads\\Advertising.csv")

print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

print("\nData types and non-null counts:")
print(df.info())

# Check for missing values — regression can't handle NaNs
print("\nMissing values per column:")
print(df.isnull().sum())

# Drop rows with missing values in our two columns of interest (if any).
# In a bigger project you might impute instead of dropping, but for a
# simple single-predictor model, dropping is usually fine.
df = df.dropna(subset=[X_COLUMN, Y_COLUMN])

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicates}")
df = df.drop_duplicates()

print(f"\nShape after cleaning: {df.shape}")


# -----------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Summary statistics: mean, std, min, max, quartiles
print("\nSummary statistics:")
print(df.describe())

# Correlation between predictor and target.
# A value close to +1 or -1 means a strong linear relationship —
# exactly what we want before fitting a linear model.
correlation = df[X_COLUMN].corr(df[Y_COLUMN])
print(f"\nCorrelation between {X_COLUMN} and {Y_COLUMN}: {correlation:.4f}")

# Scatter plot to visually inspect the relationship before modeling
plt.figure()
plt.scatter(df[X_COLUMN], df[Y_COLUMN], color="steelblue", edgecolor="white", s=70)
plt.title(f"{Y_COLUMN} vs {X_COLUMN} (Raw Data)")
plt.xlabel(X_COLUMN)
plt.ylabel(Y_COLUMN)
plt.tight_layout()
plt.savefig("01_raw_data_scatter.png", dpi=120)
plt.close()
print("\nSaved plot: 01_raw_data_scatter.png")

# Histogram of the target variable to check its distribution
plt.figure()
sns.histplot(df[Y_COLUMN], kde=True, color="darkorange")
plt.title(f"Distribution of {Y_COLUMN}")
plt.xlabel(Y_COLUMN)
plt.tight_layout()
plt.savefig("02_target_distribution.png", dpi=120)
plt.close()
print("Saved plot: 02_target_distribution.png")


# -----------------------------------------------------------------------
# 4. SPLIT DATA INTO TRAIN / TEST SETS
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: TRAIN-TEST SPLIT")
print("=" * 60)

# X must be 2D (a DataFrame of one column), y is 1D (a Series).
# Scikit-learn's fit() method expects X in shape (n_samples, n_features).
X = df[[X_COLUMN]]
y = df[Y_COLUMN]

# 80% of the data is used to train the model, 20% is held back to
# test how well the model generalizes to unseen data.
# random_state fixes the shuffle so results are reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size:     {X_test.shape[0]} samples")


# -----------------------------------------------------------------------
# 5. MODEL BUILDING & TRAINING
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: MODEL BUILDING & TRAINING")
print("=" * 60)

# Create the linear regression model object
model = LinearRegression()

# Train ("fit") the model — this is where scikit-learn calculates
# the best-fit slope and intercept using Ordinary Least Squares (OLS),
# minimizing the sum of squared errors between predictions and actuals.
model.fit(X_train, y_train)

# The learned parameters of the line:  y = (slope * x) + intercept
slope = model.coef_[0]
intercept = model.intercept_

print(f"\nModel equation:  {Y_COLUMN} = {slope:.2f} * {X_COLUMN} + {intercept:.2f}")
print(f"Slope (coefficient): {slope:.2f}")
print(f"Intercept:           {intercept:.2f}")

# Interpretation for a Sales/Newspaper example:
# - slope: each additional unit of X increases the predicted Y by this amount
# - intercept: predicted Y when X = 0


# -----------------------------------------------------------------------
# 6. MODEL EVALUATION
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: MODEL EVALUATION")
print("=" * 60)

# Generate predictions on both sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# --- Metrics on the TEST set (the one that matters most — unseen data) ---
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\nTest set performance:")
print(f"  Mean Squared Error (MSE):      {mse:,.2f}")
print(f"  Root Mean Squared Error(RMSE): {rmse:,.2f}")
print(f"  Mean Absolute Error (MAE):     {mae:,.2f}")
print(f"  R-squared (R^2):               {r2:.4f}")

# --- Compare against training set to check for overfitting ---
r2_train = r2_score(y_train, y_train_pred)
print(f"\nTrain R^2: {r2_train:.4f}  |  Test R^2: {r2:.4f}")
print("(If train R^2 is much higher than test R^2, the model may be overfitting.)")

# What these metrics mean:
# - MSE/RMSE: average squared/root-squared prediction error, in the same
#   units as the target (RMSE is easier to interpret since it's not squared).
# - MAE: average absolute error — less sensitive to outliers than MSE.
# - R^2: proportion of variance in y explained by X. Ranges up to 1.0;
#   closer to 1 means the model explains the data very well.


# -----------------------------------------------------------------------
# 7. VISUALIZATION — regression line & residuals
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: VISUALIZATION")
print("=" * 60)

# Plot 1: Regression line fitted on the TRAINING data
plt.figure()
plt.scatter(X_train, y_train, color="steelblue", label="Training data", edgecolor="white", s=70)
plt.plot(X_train, y_train_pred, color="crimson", linewidth=2, label="Regression line")
plt.title(f"{Y_COLUMN} vs {X_COLUMN} (Training Set)")
plt.xlabel(X_COLUMN)
plt.ylabel(Y_COLUMN)
plt.legend()
plt.tight_layout()
plt.savefig("03_regression_line_train.png", dpi=120)
plt.close()
print("Saved plot: 03_regression_line_train.png")

# Plot 2: Same regression line evaluated against the TEST data
# (the line's parameters come from training only — this checks generalization)
plt.figure()
plt.scatter(X_test, y_test, color="seagreen", label="Test data (actual)", edgecolor="white", s=70)
plt.plot(X_test, y_test_pred, color="crimson", linewidth=2, label="Regression line (predicted)")
plt.title(f"{Y_COLUMN} vs {X_COLUMN} (Test Set)")
plt.xlabel(X_COLUMN)
plt.ylabel(Y_COLUMN)
plt.legend()
plt.tight_layout()
plt.savefig("04_regression_line_test.png", dpi=120)
plt.close()
print("Saved plot: 04_regression_line_test.png")

# Plot 3: Residual plot — errors should scatter randomly around 0
# with no obvious pattern, which confirms a linear model was appropriate.
residuals = y_test - y_test_pred
plt.figure()
plt.scatter(y_test_pred, residuals, color="purple", edgecolor="white", s=70)
plt.axhline(y=0, color="black", linestyle="--")
plt.title("Residual Plot (Test Set)")
plt.xlabel("Predicted values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.tight_layout()
plt.savefig("05_residual_plot.png", dpi=120)
plt.close()
print("Saved plot: 05_residual_plot.png")


# -----------------------------------------------------------------------
# 8. MAKING NEW PREDICTIONS
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: PREDICTIONS ON NEW DATA")
print("=" * 60)

# Example: predict the target for a few new, unseen X values.
# Replace these with whatever inputs make sense for your dataset.
new_values = pd.DataFrame({X_COLUMN: [2, 5, 10]})
new_predictions = model.predict(new_values)

for x_val, pred in zip(new_values[X_COLUMN], new_predictions):
    print(f"  {X_COLUMN} = {x_val}  ->  predicted {Y_COLUMN} = {pred:,.2f}")

print("\nDone! Check the saved .png files for the visualizations.")
