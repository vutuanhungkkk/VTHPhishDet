from __future__ import annotations

from xgboost_pipeline import (
    DATA_PATH,
    RESULTS_DIR,
    build_feature_matrix,
    clean_data,
    evaluate_model,
    load_model,
    load_raw_data,
    maybe_save_confusion_matrix,
    split_data,
    write_metrics_summary,
)


def main():
    df = clean_data(load_raw_data(DATA_PATH))
    X, y = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = load_model()
    metrics = evaluate_model(model, X_test, y_test)

    print(metrics["classification_report"])
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    maybe_save_confusion_matrix(metrics, RESULTS_DIR)
    write_metrics_summary(metrics, RESULTS_DIR / "metrics_summary.txt", len(X_train), len(X_test))


if __name__ == "__main__":
    main()
