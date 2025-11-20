import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from fairlearn.metrics import (
    MetricFrame, true_positive_rate, false_positive_rate, selection_rate,
    equalized_odds_difference, demographic_parity_ratio, demographic_parity_difference
)

# --------------------------
# Generic Model Evaluation
# --------------------------
def evaluate_model_compas(model, X_train, X_test, y_train, y_test, X_test_orig, protected_attr, protected_group_value, scale=False):
    """
    Train model and calculate fairness metrics for COMPAS.
    - model: sklearn/XGBoost model object (not fitted)
    - scale: whether to standardize X_train/X_test (for Logistic Regression)
    """
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    sensitive_binary = (X_test_orig[protected_attr].reset_index(drop=True) == protected_group_value).astype(int)

    frame = MetricFrame(
        metrics={"accuracy": accuracy_score, "tpr": true_positive_rate, "fpr": false_positive_rate, "selection_rate": selection_rate},
        y_true=y_test.reset_index(drop=True),
        y_pred=y_pred,
        sensitive_features=sensitive_binary
    )

    return {
        "accuracy": acc,
        "tpr_difference": frame.difference(method="between_groups").get("tpr", np.nan),
        "tpr_protected": frame.by_group.get('tpr', {}).get(1, np.nan),
        "tpr_non_protected": frame.by_group.get('tpr', {}).get(0, np.nan),
        "equalized_odds": equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_binary),
        "disparate_impact": demographic_parity_ratio(y_test, y_pred, sensitive_features=sensitive_binary),
        "demographic_parity": demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_binary),
        "classification_report": report,
        "y_pred": y_pred
    }

# --------------------------
# Model Factories
# --------------------------
def decision_tree_factory():
    return DecisionTreeClassifier(random_state=42, max_depth=8)

def logistic_regression_factory():
    return LogisticRegression(max_iter=1000)

def random_forest_factory():
    return RandomForestClassifier(max_depth=4, min_samples_leaf=20, random_state=42)

def xgboost_factory():
    return XGBClassifier(max_depth=4, learning_rate=0.03, n_estimators=200,
                         subsample=0.8, colsample_bytree=0.8,
                         objective="binary:logistic", eval_metric="logloss",
                         random_state=42)

# --------------------------
# Multi-run Runner
# --------------------------
def run_multiple_model_compas(model_factory, X, y, X_orig, protected_attr, protected_group_value, n_runs=5, test_size=0.2, scale=False):
    results = []
    for _ in range(n_runs):
        X_train, X_test, y_train, y_test, _, X_test_orig = train_test_split(
            X, y, X_orig, test_size=test_size, stratify=y, random_state=None
        )

        model = model_factory()
        res = evaluate_model_compas(model, X_train, X_test, y_train, y_test, X_test_orig, protected_attr, protected_group_value, scale=scale)
        results.append(res)

    avg_results = {}
    for key in results[0]:
        if isinstance(results[0][key], (int, float, np.integer, np.floating)):
            avg_results[key] = np.mean([r[key] for r in results])
        else:
            avg_results[key] = deepcopy(results[0][key])

    return avg_results

