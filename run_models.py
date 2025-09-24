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
    """
    all_results = {name: [] for name in models.keys()}
    all_preds = {name: [] for name in models.keys()}

    for run in range(n_runs):
        for name, model_fn in models.items():
            result = evaluate_model(
                model_fn,
                X_train,
                X_test,
                y_train,
                y_test,
                X_orig,
                X_test.index
            )
            all_results[name].append(result)
            # Store the predictions for averaging later
            all_preds[name].append(result.pop('y_pred'))

    # Average numeric results across runs
    avg_results = {
        name: {
            metric: np.mean([r[metric] for r in results])
            for metric in results[0].keys()
            if metric != 'classification_report'
        }
        for name, results in all_results.items()
    }
    
    # Calculate and print a single, averaged classification report for each model
    for name, preds_list in all_preds.items():
        # Stack all predictions into a single array
        stacked_preds = np.vstack(preds_list)
        
        # Determine the most common prediction for each sample across all runs
        averaged_preds = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=stacked_preds)
        
        print(f"--- Averaged Classification Report for {name} ---")
        report_str = classification_report(y_test, averaged_preds)
        print(report_str)
        print("\n")

    return avg_results