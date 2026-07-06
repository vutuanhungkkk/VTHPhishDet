from __future__ import annotations

import argparse

from xgboost_pipeline import load_model, predict_url


def main():
    parser = argparse.ArgumentParser(description="Infer phishing probability using the trained XGBoost URL model.")
    parser.add_argument("--url", required=True, help="URL to classify.")
    args = parser.parse_args()

    model = load_model()
    result = predict_url(args.url, model)
    print(f"Prediction : {result['label']}")
    print(f"Probability: {result['probability']:.4f}")


if __name__ == "__main__":
    main()