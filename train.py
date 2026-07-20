import argparse
import pandas as pd
from src.linear_regression.model import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("--target", default="target", help="Target column name")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    model, r2 = train_model(df, args.target)
    print(f"Trained LinearRegression model — R2 on test set: {r2:.4f}")

if __name__ == "__main__":
    main()
