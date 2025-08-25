from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, selection_rate, equalized_odds_difference, demographic_parity_ratio, demographic_parity_difference
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def decision_tree(X_train, X_test, y_train, y_test, X_orig, X_test_index):
    dt = DecisionTreeClassifier(random_state=42, max_depth=8)
    dt.fit(X_train, y_train)

    y_pred_dt = dt.predict(X_test)

    acc = accuracy_score(y_test, y_pred_dt)

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
        y_pred=y_pred_dt,
        sensitive_features=sensitive_features
    )


    return {
        'accuracy': acc,
        'tpr': frame.difference(method='between_groups')['tpr'],
        'equalized_odds': equalized_odds_difference(
            y_true=y_test,
            y_pred=y_pred_dt,
            sensitive_features=sensitive_features
        ),
        'disparate_impact': demographic_parity_ratio(
            y_true=y_test,
            y_pred=y_pred_dt,
            sensitive_features=sensitive_features
        ),
        'demographic_parity': demographic_parity_difference(
            y_true=y_test,
            y_pred=y_pred_dt,
            sensitive_features=sensitive_features
        )
    }


def run_multiple_dt(X_encoded, y, sensitive_attr='sex', n_runs=5, test_size=0.2, X_orig=None):
    from sklearn.model_selection import train_test_split
    results = []

    for i in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, stratify=y
        )
        result = decision_tree(X_train, X_test, y_train, y_test, X_orig, X_test.index)
        results.append(result)

    avg_results = {key: np.mean([r[key] for r in results]) for key in results[0].keys()}
    return avg_results


def run_multiple_dt_fal(X_train, X_test, y_train, y_test, X_orig_test, protected_attr='sex', n_runs=5):
    results = []

    for _ in range(n_runs):
        result = decision_tree(
            X_train, X_test, y_train, y_test,
            X_orig=X_orig_test, X_test_index=X_test.index
        )
        results.append(result)

    avg_results = {key: np.mean([r[key] for r in results]) for key in results[0].keys()}
    return avg_results
