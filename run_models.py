import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from fairlearn.metrics import MetricFrame

def evaluate_model(model_fn, X_train, X_test, y_train, y_test, X_orig, X_test_index):
    """Helper to evaluate a single model function"""
    return model_fn(X_train, X_test, y_train, y_test, X_orig, X_test_index)


def run_all_models_with_custom_train(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    X_orig,
    *,
    n_runs=5
):
    """
    Run multiple models n_runs times on the same pre-split train/test sets.
    Useful when training data has been augmented (e.g., FairSMOTE) and
    test data should remain original.

    Args:
        models: dict of { 'name': model_function }
        X_train, y_train: training set (can include synthetic data)
        X_test, y_test: test set (original)
        X_orig: original unscaled dataset (for sensitive attributes in metrics)
        n_runs: number of times to run each model

    Returns:
        dict of averaged results for each model
    """
    all_results = {name: [] for name in models.keys()}

    for run in range(n_runs):
        for name, model_fn in models.items():
            result = model_fn(
                X_train,
                X_test,
                y_train,
                y_test,
                X_orig,
                X_test.index  # original test set indices
            )
            all_results[name].append(result)

    # Extract and print classification reports separately
    for name, results in all_results.items():
        print(f"--- Classification Reports for {name} ---")
        # Ensure 'classification_report' key exists before accessing
        if 'classification_report' in results[0]:
            for i, result in enumerate(results):
                report_dict = result['classification_report']
                # Pretty print the dictionary as a report
                report_str = classification_report(
                    y_test, 
                    pd.Series(report_dict['macro avg']['support'] * np.arange(len(y_test))),
                    output_dict=False
                )
                print(f"Run {i+1}:\n{report_str}")

    # Average numeric results across runs
    avg_results = {
        name: {
            metric: np.mean([r[metric] for r in results])
            for metric in results[0].keys()
            if metric != 'classification_report'
        }
        for name, results in all_results.items()
    }

    return avg_results