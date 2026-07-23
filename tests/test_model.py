import pandas as pd
import pytest
from src.linear_regression.model import train_model, LinearRegressionPipeline

def test_train_model_legacy():
    df = pd.DataFrame({
        "x1": [1, 2, 3, 4, 5],
        "x2": [2, 1, 0, 1, 2],
        "target": [2.5, 3.0, 3.5, 4.0, 4.5],
    })
    model, r2 = train_model(df, "target", test_size=0.4, random_state=0)
    assert hasattr(model, "coef_")
    assert isinstance(r2, float)

def test_pipeline_workflow():
    df = pd.DataFrame({
        "Hours_Studied": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "Exam_Score": [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 100.0]
    })
    
    pipeline = LinearRegressionPipeline()
    pipeline.set_data(df)

    assert "Hours_Studied" in pipeline.get_column_names()
    assert "Exam_Score" in pipeline.get_column_names()

    summary = pipeline.get_summary("Hours_Studied", "Exam_Score")
    assert summary["rows"] == 10
    assert summary["correlation"] > 0.95

    metrics = pipeline.train("Hours_Studied", "Exam_Score", test_size=0.2, random_state=42)
    assert metrics["r2"] > 0.8
    assert "slope" in metrics
    assert "intercept" in metrics

    pred = pipeline.predict_single(5.0)
    assert 40.0 <= pred <= 60.0

    fig_eda = pipeline.generate_eda_figure()
    fig_train = pipeline.generate_train_figure()
    fig_test = pipeline.generate_test_figure()

    assert fig_eda is not None
    assert fig_train is not None
    assert fig_test is not None
