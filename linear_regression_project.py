# ============================================================
# MINI PROJECT: Simple Linear Regression
# Predicting Sales based on Newspaper Advertising Budget
# ============================================================
# This project uses one input variable (Newspaper Advertising Budget) to
# predict one output variable (Sales) using Simple Linear Regression.

# ------------------------------------------------------------
# CONFIG - change these if your CSV file/column names are different
# ------------------------------------------------------------
FILE_NAME = "Advertising.csv"
INPUT_COLUMN = "Newspaper"
OUTPUT_COLUMN = "Sales"


# ------------------------------------------------------------
# STEP 1: Import the libraries we need
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ------------------------------------------------------------
# STEP 2: Load the dataset
# ------------------------------------------------------------
dataset = pd.read_csv("C:\\Users\\Ishita Bhingare\\Downloads\\Advertising.csv")

# Always good to check the real column names first, in case they
# don't match what we expect above.
print("Columns found in the file:", dataset.columns.tolist())

print("\nFirst 5 rows of the dataset:")
print(dataset.head())


# ------------------------------------------------------------
# STEP 3: Data preprocessing (cleaning the data)
# ------------------------------------------------------------
print("\nDataset info:")
print(dataset.info())

print("\nChecking for missing values:")
print(dataset.isnull().sum())

# If there are any missing values, we simply remove those rows.
# For a small, simple dataset like this, that is enough.
dataset = dataset.dropna()

# Removing duplicate rows, if any
dataset = dataset.drop_duplicates()

print("\nStatistical summary of the data:")
print(dataset.describe())


# ------------------------------------------------------------
# STEP 4: Exploratory Data Analysis (EDA)
# ------------------------------------------------------------
# Checking how strongly the two variables are related.
# A value close to 1 means a strong positive linear relationship.
correlation = dataset[INPUT_COLUMN].corr(dataset[OUTPUT_COLUMN])
print(f"\nCorrelation between {INPUT_COLUMN} and {OUTPUT_COLUMN}: {correlation:.2f}")

# A simple scatter plot to see the relationship visually before
# we even build the model.
plt.scatter(dataset[INPUT_COLUMN], dataset[OUTPUT_COLUMN], color="blue")
plt.title(f"{OUTPUT_COLUMN} vs {INPUT_COLUMN}")
plt.xlabel(INPUT_COLUMN)
plt.ylabel(OUTPUT_COLUMN)
plt.savefig("scatter_plot.png")
plt.show()


# ------------------------------------------------------------
# STEP 5: Separating input (X) and output (y)
# ------------------------------------------------------------
# X must be 2D (that is why we use double square brackets),
# y can stay 1D.
X = dataset[[INPUT_COLUMN]]
y = dataset[OUTPUT_COLUMN]


# ------------------------------------------------------------
# STEP 6: Splitting data into training set and testing set
# ------------------------------------------------------------
# We train the model on 80% of the data, and keep 20% aside to
# test how well it performs on data it has not seen before.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ------------------------------------------------------------
# STEP 7: Building and training the Simple Linear Regression model
# ------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# The model has now learned the equation: y = m*x + c
slope = model.coef_[0]
intercept = model.intercept_
print(f"\nModel equation: {OUTPUT_COLUMN} = {slope:.2f} * {INPUT_COLUMN} + {intercept:.2f}")


# ------------------------------------------------------------
# STEP 8: Making predictions on the test set
# ------------------------------------------------------------
y_pred = model.predict(X_test)

# Comparing actual vs predicted values side by side
comparison_table = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
print("\nActual vs Predicted values:")
print(comparison_table)


# ------------------------------------------------------------
# STEP 9: Evaluating the model
# ------------------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")


# ------------------------------------------------------------
# STEP 10: Visualizing the regression line (Training set)
# ------------------------------------------------------------
plt.scatter(X_train, y_train, color="blue", label="Actual (train)")
plt.plot(X_train, model.predict(X_train), color="red", label="Regression line")
plt.title(f"{OUTPUT_COLUMN} vs {INPUT_COLUMN} (Training Set)")
plt.xlabel(INPUT_COLUMN)
plt.ylabel(OUTPUT_COLUMN)
plt.legend()
plt.savefig("regression_line_training.png")
plt.show()


# ------------------------------------------------------------
# STEP 11: Visualizing the regression line (Test set)
# ------------------------------------------------------------
plt.scatter(X_test, y_test, color="green", label="Actual (test)")
plt.plot(X_train, model.predict(X_train), color="red", label="Regression line")
plt.title(f"{OUTPUT_COLUMN} vs {INPUT_COLUMN} (Test Set)")
plt.xlabel(INPUT_COLUMN)
plt.ylabel(OUTPUT_COLUMN)
plt.legend()
plt.savefig("regression_line_test.png")
plt.show()


# ------------------------------------------------------------
# STEP 12: Predicting for a brand new value
# ------------------------------------------------------------
# Example: predicting the sales for a new newspaper advertising budget of 6.5
new_input = 6.5
new_input_df = pd.DataFrame({INPUT_COLUMN: [new_input]})
predicted_output = model.predict(new_input_df)
print(f"\nPredicted {OUTPUT_COLUMN} for {INPUT_COLUMN} = {new_input}: {predicted_output[0]:.2f}")


# ------------------------------------------------------------
# STEP 13: Conclusion
# ------------------------------------------------------------
print("\n----------------------------------------------------")
print("CONCLUSION")
print("----------------------------------------------------")
print(f"The correlation between {INPUT_COLUMN} and {OUTPUT_COLUMN} is {correlation:.2f},")
print("which shows a strong linear relationship between the two variables.")
print(f"The model achieved an R2 score of {r2:.4f} on the test data, meaning it")
print("explains a large portion of the variation in the output using just one input.")
print(f"The average prediction error (RMSE) was about {rmse:.2f}, which is reasonably")
print("low compared to the overall range of the data.")
print("This confirms that Simple Linear Regression is a good fit for this dataset.")