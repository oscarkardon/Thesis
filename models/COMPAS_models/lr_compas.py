from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from fairlearn.metrics import (
    MetricFrame, true_positive_rate, false_positive_rate,
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, selection_rate
)
import numpy as np

def logistic_regression_compas(X_train, X_test, y_train, y_test, X_orig, X_test_index):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    y_pred = lr.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    sensitive_features = (X_orig.loc[X_test_index, "race"] == 0).astype(int)

    frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "selection_rate": selection_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    return {
        "accuracy": acc,
        "tpr_difference": frame.difference(method="between_groups")["tpr"],
        "tpr_non_protected": frame.by_group['tpr'].get(0, np.nan),
        "tpr_protected": frame.by_group['tpr'].get(1, np.nan),
        "equalized_odds": equalized_odds_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        ),
        "disparate_impact": demographic_parity_ratio(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        ),
        "demographic_parity": demographic_parity_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        ),
        "classification_report": report,
        "y_pred": y_pred,
    }



def run_multiple_log_reg_compas(X, y, X_orig, n_runs=5, test_size=0.2):
    from sklearn.model_selection import train_test_split
    results = []

    for _ in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=None
        )

        r = logistic_regression_compas(
            X_train, X_test, y_train, y_test,
            X_orig=X_orig, X_test_index=X_test.index
        )
        results.append(r)

    return {k: np.mean([r[k] for r in results]) for k in results[0]}
