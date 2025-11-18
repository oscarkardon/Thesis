import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def evaluate_model(model_fn, X_train, X_test, y_train, y_test, X_orig, X_test_index):
    """
    Evaluate a single model function that returns:
    {
        'accuracy': ...,
        'tpr_difference': ...,
        ...
        'classification_report': ...,
        'y_pred': array(...)
    }
    """
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
    Run multiple COMPAS-ready models n_runs times on pre-split train/test sets.
    models = {
        "Decision Tree": decision_tree_compas,
        "LogReg": logistic_regression_compas,
        ...
    }
    """

    all_results = {name: [] for name in models}
    all_preds = {name: [] for name in models}

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

            # Store metrics (minus predictions)
            all_results[name].append({
                k: v for k, v in result.items()
                if k not in ("classification_report", "y_pred")
            })

            # Store predictions for ensemble averaging later
            all_preds[name].append(result["y_pred"])

    # ----- Compute averages -----

    avg_results = {}
    for name, results in all_results.items():
        avg_results[name] = {
            metric: float(np.mean([r[metric] for r in results]))
            for metric in results[0]
        }

    # ----- Print averaged classification reports -----

    for name, preds_list in all_preds.items():

        # shape: (n_runs, n_samples)
        stacked_preds = np.vstack(preds_list)

        # majority vote across runs
        averaged_preds = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)),
            axis=0,
            arr=stacked_preds
        )

        print(f"\n--- Averaged Classification Report for {name} ---")
        print(classification_report(y_test, averaged_preds))
        print("\n")

    return avg_results
