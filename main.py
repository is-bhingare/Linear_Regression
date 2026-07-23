"""
Main Entry Point for Student Performance Analysis & Simple Linear Regression Project.
"""
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Student Performance Analysis - Linear Regression Project")
    parser.add_argument("--cli", action="store_true", help="Run project in CLI mode instead of GUI")
    parser.add_argument("--csv", default="Advertising.csv", help="Path to CSV dataset for CLI mode")
    parser.add_argument("--target", default="Sales", help="Target column for CLI mode")
    args = parser.parse_args()

    if args.cli:
        from src.linear_regression.model import LinearRegressionPipeline
        print(f"Running Linear Regression in CLI mode on: {args.csv}")
        pipeline = LinearRegressionPipeline()
        pipeline.load_csv(args.csv)
        cols = pipeline.get_column_names()
        if "Newspaper" in cols and args.target == "Sales":
            feature_col = "Newspaper"
        else:
            valid_cols = [c for c in cols if c != args.target and not c.startswith("Unnamed")]
            feature_col = valid_cols[0] if valid_cols else [c for c in cols if c != args.target][0]
        metrics = pipeline.train(feature_col, args.target)

        print("\n=== Model Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")
    else:
        from gui_app import StudentRegressionApp
        print("Launching Desktop GUI Application...")
        app = StudentRegressionApp()
        app.mainloop()

if __name__ == "__main__":
    main()
