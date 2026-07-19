from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(test_size: float = 0.2, random_state: int = 42):
    dataset = load_diabetes(as_frame=True)
    data = dataset.frame

    x = data[["bmi"]]
    y = data["target"]

    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def train_model(model: Pipeline, x_train, y_train) -> Pipeline:
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: Pipeline, x_test, y_test) -> dict:
    predictions = model.predict(x_test)
    return {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


def plot_regression_line(model: Pipeline, x, y, output_path: Path) -> None:
    x_values = np.asarray(x).ravel()
    y_values = np.asarray(y).ravel()

    order = np.argsort(x_values)
    sorted_x = x_values[order]

    regression_inputs = pd.DataFrame(sorted_x, columns=["bmi"])
    regression_line = model.predict(regression_inputs)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, y_values, alpha=0.6, label="Data points")
    plt.plot(sorted_x, regression_line, color="red", label="Regression line")
    plt.xlabel("BMI (normalized)")
    plt.ylabel("Target")
    plt.title("Linear Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    x_train, x_test, y_train, y_test = load_and_preprocess_data()

    model = build_model()
    trained_model = train_model(model, x_train, y_train)

    metrics = evaluate_model(trained_model, x_test, y_test)
    print("Model evaluation metrics:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")

    output_file = Path(__file__).resolve().parent / "regression_plot.png"
    full_x = np.vstack([x_train.values, x_test.values]).ravel()
    full_y = np.concatenate([y_train.values, y_test.values])

    plot_regression_line(trained_model, full_x, full_y, output_file)
    print(f"Saved visualization to: {output_file}")


if __name__ == "__main__":
    main()
