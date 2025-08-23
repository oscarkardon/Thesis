from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def random_forest_model(X_train, X_test, y_train, y_test, X_orig, X_test_index):
    rf = RandomForestClassifier(max_depth=4, min_samples_leaf=20, random_state=42)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred_rf)
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
        y_pred=y_pred_rf,
        sensitive_features=sensitive_features
    )


    return {
        'accuracy': acc,
        'tpr': frame.difference(method='between_groups')['tpr'],
        'equalized_odds': equalized_odds_difference(
            y_true=y_test,
            y_pred=y_pred_rf,
            sensitive_features=sensitive_features
        ),
        'disparate_impact': demographic_parity_ratio(
            y_true=y_test,
            y_pred=y_pred_rf,
            sensitive_features=sensitive_features
        ),
        'demographic_parity': demographic_parity_difference(
            y_true=y_test,
            y_pred=y_pred_rf,
            sensitive_features=sensitive_features
        )
    }

def run_multiple_rf(X, y, sensitive_attr='sex', n_runs=5, test_size=0.2, X_orig=None):
    from sklearn.model_selection import train_test_split

    results = []

    for i in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y
        )
        result = random_forest_model(X_train, X_test, y_train, y_test, X_orig, X_test.index)
        results.append(result)

    avg_results = {key: np.mean([r[key] for r in results]) for key in results[0].keys()}
    return avg_results
