import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from fairlearn.metrics import (
    MetricFrame, true_positive_rate, false_positive_rate, selection_rate,
    equalized_odds_difference, demographic_parity_ratio, demographic_parity_difference
)

# --------------------------
# Core Fairness Logic
# --------------------------
def _compute_metrics(y_true, y_pred, X_test_orig, protected_attr, protected_group_value):
    """Internal helper to calculate fairness metrics."""
    sensitive_binary = (X_test_orig[protected_attr].reset_index(drop=True) == protected_group_value).astype(int)
    
    frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score, 
            "tpr": true_positive_rate, 
            "fpr": false_positive_rate, 
            "selection_rate": selection_rate
        },
        y_true=y_true.reset_index(drop=True),
        y_pred=y_pred,
        sensitive_features=sensitive_binary
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "tpr_difference": frame.difference(method="between_groups").get("tpr", np.nan),
        "tpr_protected": frame.by_group.get('tpr', {}).get(1, np.nan),
        "tpr_non_protected": frame.by_group.get('tpr', {}).get(0, np.nan),
        "equalized_odds": equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_binary),
        "disparate_impact": demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_binary),
        "demographic_parity": demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_binary),
        "y_pred": y_pred
    }

# --------------------------
# Model Wrappers (The model_fn objects)
# --------------------------
def train_dt(X_train, X_test, y_train, y_test, X_test_orig, attr, val):
    model = DecisionTreeClassifier(random_state=42, max_depth=8)
    model.fit(X_train, y_train)
    return _compute_metrics(y_test, model.predict(X_test), X_test_orig, attr, val)

def train_lr(X_train, X_test, y_train, y_test, X_test_orig, attr, val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)
    return _compute_metrics(y_test, model.predict(X_test_s), X_test_orig, attr, val)

def train_rf(X_train, X_test, y_train, y_test, X_test_orig, attr, val):
    model = RandomForestClassifier(max_depth=4, min_samples_leaf=20, random_state=42)
    model.fit(X_train, y_train)
    return _compute_metrics(y_test, model.predict(X_test), X_test_orig, attr, val)

def train_xgb(X_train, X_test, y_train, y_test, X_test_orig, attr, val):
    model = XGBClassifier(max_depth=4, learning_rate=0.03, n_estimators=200, random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)
    return _compute_metrics(y_test, model.predict(X_test), X_test_orig, attr, val)

# --------------------------
# Generic Model Runner
# --------------------------
def evaluate_model(model_fn, X_train, X_test, y_train, y_test, X_test_orig, protected_attr, protected_group_value):
    return model_fn(X_train, X_test, y_train, y_test, X_test_orig, protected_attr, protected_group_value)

def run_all_models_income(
    models_dict, X_train, y_train, X_test, y_test, X_test_orig, 
    protected_attr, protected_group_value, n_runs=5
):
    all_results = {}
    
    for name, model_fn in models_dict.items():
        results_list = []
        preds_list = []
        
        for _ in range(n_runs):
            res = evaluate_model(model_fn, X_train, X_test, y_train, y_test, X_test_orig, protected_attr, protected_group_value)
            results_list.append({k: v for k, v in res.items() if k != 'y_pred'})
            preds_list.append(res['y_pred'])

        # Average numeric metrics
        avg_res = {}
        for key in results_list[0]:
            if isinstance(results_list[0][key], (int, float, np.integer, np.floating)):
                avg_res[key] = np.mean([r[key] for r in results_list])
            else:
                avg_res[key] = deepcopy(results_list[0][key])

        # Majority Vote for Classification Report
        stacked_preds = np.vstack(preds_list)
        averaged_preds = np.apply_along_axis(lambda x: np.argmax(np.bincount(x.astype(int))), axis=0, arr=stacked_preds)
        
        print(f"--- Averaged Classification Report for {name} ---")
        print(classification_report(y_test, averaged_preds))
        
        all_results[name] = avg_res

    return all_results