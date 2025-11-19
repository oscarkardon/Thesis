from fairlearn.metrics import (
    MetricFrame, true_positive_rate, false_positive_rate, selection_rate,
    equalized_odds_difference, demographic_parity_ratio, demographic_parity_difference
)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def decision_tree_compas(X_train, X_test, y_train, y_test, X_orig, X_test_index):
    """Decision tree fairness evaluation for COMPAS."""

    dt = DecisionTreeClassifier(random_state=42, max_depth=8)
    dt.fit(X_train, y_train)

    # store predictions
    y_pred_dt = dt.predict(X_test)

    acc = accuracy_score(y_test, y_pred_dt)
    report = classification_report(y_test, y_pred_dt, output_dict=True)

    # Protected group = race==0
    sensitive_features = (X_orig.loc[X_test_index, "race"] == 0).astype(int)

    frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "selection_rate": selection_rate
        },
        y_true=y_test,
        y_pred=y_pred_dt,          # FIXED
        sensitive_features=sensitive_features
    )

    # Use .get() to avoid KeyError if a group is missing
    tpr_non_protected = frame.by_group['tpr'].get(0, np.nan)
    tpr_protected = frame.by_group['tpr'].get(1, np.nan)

    return {
        "accuracy": acc,
        "tpr_difference": frame.difference(method="between_groups")["tpr"],
        "tpr_non_protected": tpr_non_protected,
        "tpr_protected": tpr_protected,
        "equalized_odds": equalized_odds_difference(
            y_true=y_test,
            y_pred=y_pred_dt,
            sensitive_features=sensitive_features
        ),
        "disparate_impact": demographic_parity_ratio(
            y_true=y_test,
            y_pred=y_pred_dt,
            sensitive_features=sensitive_features
        ),
        "demographic_parity": demographic_parity_difference(
            y_true=y_test,
            y_pred=y_pred_dt,
            sensitive_features=sensitive_features
        ),
        "classification_report": report,
        "y_pred": y_pred_dt
    }



def run_multiple_dt_compas(X, y, n_runs=5, test_size=0.2, X_orig=None):
    """Runs COMPAS DT model multiple times and averages results."""
    from sklearn.model_selection import train_test_split
    results = []

    for _ in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=None
        )

        res = decision_tree_compas(
            X_train, X_test, y_train, y_test,
            X_orig=X_orig,
            X_test_index=X_test.index
        )
        results.append(res)

    avg_results = {k: np.mean([r[k] for r in results]) for k in results[0]}
    return avg_results


def run_multiple_dt_fal_compas(X_train, X_test, y_train, y_test, X_orig_test, n_runs=5):
    """Runs COMPAS DT model multiple times in FAL loop."""
    results = []

    for _ in range(n_runs):
        res = decision_tree_compas(
            X_train, X_test, y_train, y_test,
            X_orig=X_orig_test,
            X_test_index=X_test.index
        )
        results.append(res)

    avg_results = {k: np.mean([r[k] for r in results]) for k in results[0]}
    return avg_results
