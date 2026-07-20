# Linear_Regression

Minimal Python project demonstrating training a scikit-learn LinearRegression model.

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
