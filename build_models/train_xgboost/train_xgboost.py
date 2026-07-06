from __future__ import annotations

from xgboost_pipeline import (
    DATA_PATH,
    MODEL_PATH,
    RESULTS_DIR,
    build_feature_matrix,
    build_model,
    clean_data,
    evaluate_model,
    load_raw_data,
    maybe_save_confusion_matrix,
    save_model,
    split_data,
    write_metrics_summary,
)


def main():
    df = clean_data(load_raw_data(DATA_PATH))
    X, y = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    scale_pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    model = build_model(scale_pos_weight)
    model.fit(X_train, y_train)
    print("Model trained ✅")

    save_model(model, MODEL_PATH)
    metrics = evaluate_model(model, X_test, y_test)
    print(metrics["classification_report"])

    maybe_save_confusion_matrix(metrics, RESULTS_DIR)
    write_metrics_summary(metrics, RESULTS_DIR / "metrics_summary.txt", len(X_train), len(X_test))
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
