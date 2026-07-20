# Linear_Regression

This project involves building a simple linear regression model to predict a target variable based on a single predictor variable. It demonstrates training and evaluating a scikit-learn `LinearRegression` model and includes a small example runner and tests.

Getting started:

1. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the example training script:

```bash
python train.py path/to/your.csv --target target_column_name
```

Run tests:

```bash
python -m pytest -q
```

Files of interest:

- `train.py` - small runner to train on a CSV
- `src/linear_regression/model.py` - training logic
- `requirements.txt` - dependencies

