import pandas as pd
from src.linear_regression.model import train_model

def test_train_model():
    df = pd.DataFrame({
        "x1": [1, 2, 3, 4, 5],
        "x2": [2, 1, 0, 1, 2],
        "target": [2.5, 3.0, 3.5, 4.0, 4.5],
    })
    model, r2 = train_model(df, "target", test_size=0.4, random_state=0)
    assert hasattr(model, "coef_")
    assert isinstance(r2, float)
