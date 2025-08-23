from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    MetricFrame,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    equalized_odds_ratio,
    equalized_odds_difference,
    demographic_parity_ratio,
    selection_rate
)

def logistic_regression(X_train, X_test, y_train, y_test, X_orig, X_test_index):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    y_pred_log = log_reg.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred_log)

    # sensitive feature (sex) from original unscaled data
    sensitive_features = X_orig.loc[X_test_index, 'sex']

    # Fairlearn MetricFrame
    frame = MetricFrame(
        metrics={
            'accuracy': accuracy_score,
            'tpr': true_positive_rate,
            'fpr': false_positive_rate,
            'selection_rate': selection_rate,
        },
        y_true=y_test,
        y_pred=y_pred_log,
        sensitive_features=sensitive_features
    )


    return {
        'accuracy': acc,
        'tpr': frame.difference(method='between_groups')['tpr'],
        'equalized_odds': equalized_odds_difference(
            y_true=y_test,
            y_pred=y_pred_log,
            sensitive_features=sensitive_features
        ),
        'disparate_impact': demographic_parity_ratio(
            y_true=y_test,
            y_pred=y_pred_log,
            sensitive_features=sensitive_features
        ),
        'demographic_parity': demographic_parity_difference(
            y_true=y_test,
            y_pred=y_pred_log,
            sensitive_features=sensitive_features
        )
}

import numpy as np
from sklearn.model_selection import train_test_split

def run_multiple_log_reg(X, y, X_orig, n_runs=5, test_size=0.2):
    all_results = []

    for _ in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=None
        )

        result = logistic_regression(
            X_train, X_test, y_train, y_test,
            X_orig=X_orig, X_test_index=X_test.index
        )
        all_results.append(result)

    # Average the metrics
    averaged = {
        k: np.nanmean([r[k] for r in all_results])
        for k in all_results[0]
    }

    return averaged
