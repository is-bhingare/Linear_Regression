# 🎓 Student Performance Analysis Studio — Linear Regression Project

A comprehensive, interactive Desktop GUI Application built with **Python, Tkinter, and Matplotlib** for predicting student exam performance (or marketing sales) using **Simple Linear Regression**.

Designed specifically for **College Mini-Project Demonstrations**, this application replaces terminal-only scripts with a clean split-pane window, single-window tabbed graph viewer, interactive action controls, and an output console logging statistics, metrics, and predictions.

---

## 🌟 Key Features

1. **Interactive Desktop GUI Interface**:
   - Modern styled toolbar with buttons: **`Load Dataset`**, **`Show EDA`**, **`Train Model`**, **`View Graphs`**, **`Show Results`**, and **`Predict`**.
   - Input Feature ($X$) and Target Output ($y$) dropdown selectors with automatic column detection.

2. **Single-Window Embedded Graph Viewer**:
   - Displays all visualizations inside a single window using `ttk.Notebook` tabs and Next/Previous navigation buttons:
     - **Tab 1: EDA Scatter Plot** — Exploratory scatter plot of raw data distribution.
     - **Tab 2: Training Set Regression Line** — Actual training data points overlaid with fitted linear regression line ($y = mx + c$).
     - **Tab 3: Test Set & Residuals Plot** — Actual test points vs model predictions with residual error links.

3. **Output & Log Panel**:
   - Dedicated scrollable text console replacing terminal output.
   - Logs dataset information, head preview, null checks, summary statistics (`describe()`), correlation coefficient ($r$), regression equation, and evaluation metrics ($MSE$, $RMSE$, $MAE$, $R^2$).

4. **Live Prediction Dock & Dialog**:
   - Interactive prediction field allowing users to input feature values (e.g. Study Hours = `8.5`) and instantly calculate predicted scores (e.g., Exam Score = `92.45`) with detailed calculation steps.

5. **Sample Datasets Included**:
   - Bundled with `student_scores.csv` (**Study Hours vs Exam Score**) for immediate out-of-the-box college project demonstration, as well as support for custom CSV datasets (e.g. `Advertising.csv`).

---

## 🛠️ Quick Start & Usage

### 1. Environment Setup

Activate your Python virtual environment:
```powershell
# Windows PowerShell
.\.venv\Scripts\activate
```

### 2. Launch the Desktop GUI

```powershell
python main.py
```
*(or run `python gui_app.py` directly)*

### 3. (Optional) Run CLI Mode
```powershell
python main.py --cli --csv student_scores.csv --target Exam_Score
```

### 4. Run Unit Tests
```powershell
python -m pytest
```

---

## 📁 Project Structure

- `gui_app.py` — Main Tkinter Desktop Application with embedded Matplotlib tabs & output panel.
- `main.py` — Application launcher supporting GUI and CLI modes.
- `student_scores.csv` — Default sample dataset for Student Performance Analysis.
- `src/linear_regression/model.py` — Core ML pipeline (`LinearRegressionPipeline`) for data preprocessing, training, evaluation metrics, and figure generation.
- `tests/test_model.py` — Unit tests for model pipeline.
- `requirements.txt` — Project dependencies (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).

---

## 🎓 College Mini-Project Demonstration Workflow

1. Launch GUI via `python main.py`.
2. Click **`Show EDA`** to demonstrate data distribution, missing value checks, and correlation statistics ($r \approx 0.95$).
3. Click **`Train Model`** to execute train/test split (80/20) and display regression parameters ($m$, $c$) and evaluation metrics ($R^2$, $RMSE$).
4. Click **`View Graphs`** and use **`Next Graph ⏭`** / **`⏮ Previous Graph`** or Notebook tabs to showcase embedded plots without opening separate windows.
5. Click **`Show Results`** to view actual vs predicted comparisons and project conclusion.
6. Enter an input value in **Quick Prediction** (e.g., `8.5` Study Hours) and click **`Calculate Prediction`** to demonstrate real-time prediction.
