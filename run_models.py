import numpy as np
from sklearn.model_selection import train_test_split

def evaluate_model(model_fn, X_train, X_test, y_train, y_test, X_orig, X_test_index):
    """Helper to evaluate a single model function"""
    return model_fn(X_train, X_test, y_train, y_test, X_orig, X_test_index)


def run_all_models(models, X_encoded, y, X_orig, sensitive_attr='sex', n_runs=5, test_size=0.2, random_state=42):
    """
    Run multiple models n_runs times with the same train/test split each run.
    
    Args:
        models: dict of { 'name': model_function }
        X_encoded: features
        y: labels
        X_orig: original unscaled DataFrame with sensitive attributes
        sensitive_attr: column name of sensitive attribute
        n_runs: number of runs
        test_size: test split size
        random_state: for reproducibility
    
    Returns:
        dict of averaged results for each model
    """
    all_results = {name: [] for name in models.keys()}

    for run in range(n_runs):
        # One split per run, shared across all models
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, stratify=y, random_state=random_state + run
        )
        
        for name, model_fn in models.items():
            result = evaluate_model(model_fn, X_train, X_test, y_train, y_test, X_orig, X_test.index)
            all_results[name].append(result)

    # Average results
    avg_results = {
        name: {metric: np.mean([r[metric] for r in results]) for metric in results[0].keys()}
        for name, results in all_results.items()
    }

    return avg_results



def run_all_models_with_custom_train(models, X_encoded, y, X_orig, protected_attr='sex', n_runs=5, test_size=0.2, random_state=42):
    """
    Run multiple models n_runs times using pre-defined training data (which could already include synthetic samples),
    while keeping test data original. Does NOT generate synthetic data itself.

    Args:
        models: dict of {'name': model_function}
        X_encoded: features (numeric, preprocessed)
        y: labels
        X_orig: original unscaled DataFrame (for sensitive features in metrics)
        protected_attr: column name of sensitive attribute
        n_runs: number of runs
        test_size: fraction of data for test
        random_state: for reproducibility

    Returns:
        dict of averaged results for each model
    """
    all_results = {name: [] for name in models.keys()}

    for run in range(n_runs):
        # Split train/test for this run
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, stratify=y, random_state=random_state + run
        )

        # Evaluate each model with given train/test split
        for name, model_fn in models.items():
            result = model_fn(X_train, X_test, y_train, y_test, X_orig=X_orig, X_test_index=X_test.index)
            all_results[name].append(result)

    # Average results across runs
    avg_results = {
        name: {metric: np.mean([r[metric] for r in results]) for metric in results[0].keys()}
        for name, results in all_results.items()
    }

    return avg_results
