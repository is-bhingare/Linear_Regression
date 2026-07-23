import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for thread-safe plot generation
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_model(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """Backward-compatible train_model function."""
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return model, r2


class LinearRegressionPipeline:
    """Complete Simple/Multiple Linear Regression pipeline with graph generation for GUI embedding."""

    def __init__(self):
        self.df = None
        self.feature_col = None
        self.target_col = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.metrics = {}
        self.comparison_df = None

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV data and perform basic cleaning."""
        self.df = pd.read_csv(file_path)
        self.df = self.df.dropna().drop_duplicates()
        return self.df

    def set_data(self, df: pd.DataFrame):
        """Set dataframe directly."""
        self.df = df.dropna().drop_duplicates()

    def get_column_names(self):
        if self.df is None:
            return []
        return list(self.df.columns)

    def get_summary(self, feature_col: str, target_col: str) -> dict:
        """Return basic exploratory data stats and correlation."""
        if self.df is None:
            raise ValueError("No dataset loaded.")
        
        corr = self.df[feature_col].corr(self.df[target_col])
        describe_df = self.df[[feature_col, target_col]].describe()
        
        return {
            "rows": len(self.df),
            "columns": list(self.df.columns),
            "head": self.df.head(5),
            "describe": describe_df,
            "null_count": self.df[[feature_col, target_col]].isnull().sum().to_dict(),
            "correlation": corr
        }

    def train(self, feature_col: str, target_col: str, test_size: float = 0.2, random_state: int = 42):
        """Train simple linear regression model and compute evaluation metrics."""
        if self.df is None:
            raise ValueError("No dataset loaded.")

        self.feature_col = feature_col
        self.target_col = target_col

        X = self.df[[feature_col]]
        y = self.df[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)

        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        slope = float(self.model.coef_[0])
        intercept = float(self.model.intercept_)

        self.metrics = {
            "slope": slope,
            "intercept": intercept,
            "equation": f"{target_col} = {slope:.2f} * {feature_col} + {intercept:.2f}",
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "train_samples": len(self.X_train),
            "test_samples": len(self.X_test)
        }

        self.comparison_df = pd.DataFrame({
            "Actual": self.y_test.values,
            "Predicted": self.y_pred,
            "Error (Residual)": self.y_test.values - self.y_pred
        })

        return self.metrics

    def predict_single(self, input_val: float) -> float:
        """Predict target value for a single feature value."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        input_df = pd.DataFrame({self.feature_col: [input_val]})
        pred = self.model.predict(input_df)[0]
        return float(pred)

    def generate_eda_figure(self, feature_col: str = None, target_col: str = None) -> plt.Figure:
        """Generate Matplotlib figure for EDA scatter plot."""
        feat = feature_col or self.feature_col
        targ = target_col or self.target_col
        if not feat or not targ:
            raise ValueError("Feature and target columns must be specified.")

        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=100)
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#ffffff")

        ax.scatter(self.df[feat], self.df[targ], color="#2b5c8f", alpha=0.8, edgecolors="none", s=50)
        ax.set_title(f"EDA: {targ} vs {feat}", fontsize=12, fontweight="bold", color="#1f2937")
        ax.set_xlabel(feat, fontsize=10, fontweight="bold", color="#374151")
        ax.set_ylabel(targ, fontsize=10, fontweight="bold", color="#374151")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        return fig

    def generate_train_figure(self) -> plt.Figure:
        """Generate Matplotlib figure for Training set regression line."""
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=100)
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#ffffff")

        ax.scatter(self.X_train[self.feature_col], self.y_train, color="#2563eb", label="Train Data (Actual)", s=45, alpha=0.7)
        
        # Sort X_train for smooth line plot
        X_sorted = self.X_train.sort_values(by=self.feature_col)
        y_line = self.model.predict(X_sorted)
        ax.plot(X_sorted[self.feature_col], y_line, color="#dc2626", linewidth=2.5, label="Regression Fit Line")

        ax.set_title(f"Training Set: Fit Line ({self.metrics.get('equation', '')})", fontsize=11, fontweight="bold", color="#1f2937")
        ax.set_xlabel(self.feature_col, fontsize=10, fontweight="bold", color="#374151")
        ax.set_ylabel(self.target_col, fontsize=10, fontweight="bold", color="#374151")
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        return fig

    def generate_test_figure(self) -> plt.Figure:
        """Generate Matplotlib figure for Test set predictions & residuals."""
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=100)
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#ffffff")

        ax.scatter(self.X_test[self.feature_col], self.y_test, color="#059669", label="Test Data (Actual)", s=55, zorder=3)
        ax.scatter(self.X_test[self.feature_col], self.y_pred, color="#d97706", marker="x", s=65, label="Test Predictions", zorder=4)

        # Plot regression line across range
        X_range = pd.DataFrame({self.feature_col: np.linspace(self.df[self.feature_col].min(), self.df[self.feature_col].max(), 100)})
        y_range_pred = self.model.predict(X_range)
        ax.plot(X_range[self.feature_col], y_range_pred, color="#dc2626", linestyle="-", linewidth=2, label="Model Line", zorder=2)

        # Connect actual and predicted points with residual lines
        for x_val, y_act, y_pr in zip(self.X_test[self.feature_col], self.y_test, self.y_pred):
            ax.plot([x_val, x_val], [y_act, y_pr], color="#9ca3af", linestyle=":", linewidth=1.2)

        ax.set_title(f"Test Set Evaluation ($R^2$: {self.metrics.get('r2', 0):.4f})", fontsize=11, fontweight="bold", color="#1f2937")
        ax.set_xlabel(self.feature_col, fontsize=10, fontweight="bold", color="#374151")
        ax.set_ylabel(self.target_col, fontsize=10, fontweight="bold", color="#374151")
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        return fig
