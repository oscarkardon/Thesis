import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from fairlearn.metrics import (
    MetricFrame, true_positive_rate, false_positive_rate, selection_rate,
    equalized_odds_difference, demographic_parity_ratio, demographic_parity_difference
)


# --------------------------
# Decision Tree Evaluation
# --------------------------
def decision_tree_compas(X_train, X_test, y_train, y_test, X_test_orig, protected_attr, protected_group_value):
    """
    Train a Decision Tree and compute fairness metrics.
    """
    dt = DecisionTreeClassifier(random_state=42, max_depth=8)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Sensitive attribute
    sensitive_features = X_test_orig[protected_attr].reset_index(drop=True)
    sensitive_binary = (sensitive_features == protected_group_value).astype(int)

    frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "selection_rate": selection_rate
        },
        y_true=y_test.reset_index(drop=True),
        y_pred=y_pred,
        sensitive_features=sensitive_binary
    )

    tpr_by_group = frame.by_group.get('tpr', {})
    tpr_protected = tpr_by_group.get(1, np.nan)
    tpr_non_protected = tpr_by_group.get(0, np.nan)

    return {
        "accuracy": acc,
        "tpr_difference": frame.difference(method="between_groups").get("tpr", np.nan),
        "tpr_protected": tpr_protected,
        "tpr_non_protected": tpr_non_protected,
        "equalized_odds": equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_binary),
        "disparate_impact": demographic_parity_ratio(y_test, y_pred, sensitive_features=sensitive_binary),
        "demographic_parity": demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_binary),
        "classification_report": report,
        "y_pred": y_pred
    }


# --------------------------
# Run Multiple Random Splits
# --------------------------
def run_multiple_dt_compas(X, y, X_orig, protected_attr, protected_group_value, n_runs=5, test_size=0.2):
    results = []
    for _ in range(n_runs):
        X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = train_test_split(
            X, y, X_orig, test_size=test_size, stratify=y, random_state=None
        )

        res = decision_tree_compas(
            X_train, X_test, y_train, y_test,
            X_test_orig=X_test_orig,
            protected_attr=protected_attr,
            protected_group_value=protected_group_value
        )
        results.append(res)

    avg_results = {}
    for key in results[0]:
        if isinstance(results[0][key], (int, float, np.floating, np.integer)):
            avg_results[key] = np.mean([r[key] for r in results])
        else:
            avg_results[key] = deepcopy(results[0][key])

    return avg_results


# --------------------------
# Run Multiple FAL Iterations (No Random Split)
# --------------------------
def run_multiple_dt_fal_compas(X_train, X_test, y_train, y_test, X_test_orig, protected_attr, protected_group_value, n_runs=5):
    results = []
    for _ in range(n_runs):
        res = decision_tree_compas(
            X_train, X_test, y_train, y_test,
            X_test_orig=X_test_orig,
            protected_attr=protected_attr,
            protected_group_value=protected_group_value
        )
        results.append(res)

    avg_results = {}
    for key in results[0]:
        if isinstance(results[0][key], (int, float, np.floating, np.integer)):
            avg_results[key] = np.mean([r[key] for r in results])
        else:
            avg_results[key] = deepcopy(results[0][key])

    return avg_results
