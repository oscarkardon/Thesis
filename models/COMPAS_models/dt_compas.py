from fairlearn.metrics import (
    MetricFrame, true_positive_rate, false_positive_rate, selection_rate,
    equalized_odds_difference, demographic_parity_ratio, demographic_parity_difference
)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from copy import deepcopy

def decision_tree_compas(X_train, X_test, y_train, y_test, X_test_orig, protected_attr, protected_group_value):
    """
    Train a decision tree and compute fairness metrics.
    - X_test_orig: original test features (unscaled) to extract protected attribute
    - protected_attr: column name of protected attribute
    - protected_group_value: value indicating the protected group (e.g., 0 or 'African-American')
    """
    
    # --- Train Decision Tree ---
    dt = DecisionTreeClassifier(random_state=42, max_depth=8)
    dt.fit(X_train, y_train)
    
    # --- Predictions ---
    y_pred = dt.predict(X_test)
    
    # --- Accuracy & Classification Report ---
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # --- Sensitive Feature Alignment ---
    sensitive_features = X_test_orig[protected_attr].reset_index(drop=True)
    
    # Convert to 0/1 coding for MetricFrame: 1 = protected group, 0 = non-protected
    sensitive_binary = (sensitive_features == protected_group_value).astype(int)
    
    # --- MetricFrame for group metrics ---
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
    
    # Handle missing groups gracefully
    tpr_by_group = frame.by_group.get('tpr', {})
    tpr_protected = tpr_by_group.get(1, np.nan)
    tpr_non_protected = tpr_by_group.get(0, np.nan)
    
    # --- Fairness metrics ---
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_binary)
    dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sensitive_binary)
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_binary)
    
    return {
        "accuracy": acc,
        "tpr_difference": frame.difference(method="between_groups").get("tpr", np.nan),
        "tpr_protected": tpr_protected,
        "tpr_non_protected": tpr_non_protected,
        "equalized_odds": eo_diff,
        "disparate_impact": dp_ratio,
        "demographic_parity": dp_diff,
        "classification_report": report,
        "y_pred": y_pred
    }


def run_multiple_dt_compas(X, y, protected_attr, protected_group_value, X_orig=None, n_runs=5, test_size=0.2):
    """
    Run decision tree multiple times with random splits and average scalar metrics.
    """
    from sklearn.model_selection import train_test_split
    
    results = []
    
    for i in range(n_runs):
        # Split with stratification to preserve label distribution
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
    
    # --- Average only scalar metrics ---
    avg_results = {}
    for key in results[0]:
        if isinstance(results[0][key], (int, float, np.floating, np.integer)):
            avg_results[key] = np.mean([r[key] for r in results])
        else:
            avg_results[key] = deepcopy(results[0][key])  # Keep one example for dict-like metrics
    
    return avg_results


def run_multiple_dt_fal_compas(X_train, X_test, y_train, y_test, X_test_orig, protected_attr, protected_group_value, n_runs=5):
    """
    Run decision tree multiple times in a FAL loop (no random splitting).
    """
    results = []
    for _ in range(n_runs):
        res = decision_tree_compas(
            X_train, X_test, y_train, y_test,
            X_test_orig=X_test_orig,
            protected_attr=protected_attr,
            protected_group_value=protected_group_value
        )
        results.append(res)
    
    # Average scalar metrics only
    avg_results = {}
    for key in results[0]:
        if isinstance(results[0][key], (int, float, np.floating, np.integer)):
            avg_results[key] = np.mean([r[key] for r in results])
        else:
            avg_results[key] = deepcopy(results[0][key])
    
    return avg_results
