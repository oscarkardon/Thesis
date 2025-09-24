import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def evaluate_model(model_fn, X_train, X_test, y_train, y_test, X_orig, X_test_index):
    """Helper to evaluate a single model function"""
    return model_fn(X_train, X_test, y_train, y_test, X_orig, X_test_index)



def run_all_models(models, X_encoded, y, X_orig, sensitive_attr='sex', n_runs=5, test_size=0.2, random_state=42):
    # This code remains the same as your original, but it will need to be updated to capture the new 'classification_report' entry in the returned dictionary
    all_results = {name: [] for name in models.keys()}

    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, stratify=y, random_state=random_state + run
        )
        
        for name, model_fn in models.items():
            result = evaluate_model(model_fn, X_train, X_test, y_train, y_test, X_orig, X_test.index)
            all_results[name].append(result)

    # Extract and print classification reports separately
    for name, results in all_results.items():
        print(f"--- Classification Reports for {name} ---")
        for i, result in enumerate(results):
            print(f"Run {i+1}:\n{classification_report(y_test, result['classification_report'], output_dict=False)}")
            
    # Average numeric results
    avg_results = {
        name: {metric: np.mean([r[metric] for r in results]) for metric in results[0].keys() if metric != 'classification_report'}
        for name, results in all_results.items()
    }
    
    return avg_results


def run_all_models_with_custom_train(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    X_orig,
    *,
    n_runs=5
):
    all_results = {name: [] for name in models.keys()}

    for run in range(n_runs):
        for name, model_fn in models.items():
            result = model_fn(
                X_train,
                X_test,
                y_train,
                y_test,
                X_orig,
                X_test.index
            )
            all_results[name].append(result)

    # Extract and print classification reports separately
    for name, results in all_results.items():
        print(f"--- Classification Reports for {name} ---")
        for i, result in enumerate(results):
            print(f"Run {i+1}:\n{classification_report(y_test, result['classification_report'], output_dict=False)}")
            
    # Average numeric results
    avg_results = {
        name: {metric: np.mean([r[metric] for r in results]) for metric in results[0].keys() if metric != 'classification_report'}
        for name, results in all_results.items()
    }
    return avg_results