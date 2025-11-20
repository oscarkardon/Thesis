# --------------------------
# Generic Model Runner
# --------------------------
def evaluate_model(model_fn, X_train, X_test, y_train, y_test, X_test_orig, protected_attr, protected_group_value):
    """Helper to evaluate a single model function with protected attribute."""
    return model_fn(X_train, X_test, y_train, y_test, X_test_orig, protected_attr, protected_group_value)


def run_all_models_with_custom_train(
    models,
    X_train, y_train, X_test, y_test, X_test_orig,
    protected_attr, protected_group_value,
    n_runs=5
):
    all_results = {}
    all_preds = {}

    for name, model_fn in models.items():
        results_list = []
        preds_list = []
        for _ in range(n_runs):
            res = evaluate_model(
                model_fn,
                X_train, X_test, y_train, y_test,
                X_test_orig,
                protected_attr,
                protected_group_value
            )
            results_list.append({k: v for k, v in res.items() if k != 'y_pred'})
            preds_list.append(res['y_pred'])
        all_results[name] = results_list
        all_preds[name] = preds_list

    # Average numeric metrics
    avg_results = {}
    for name in all_results:
        avg_results[name] = {}
        for key in all_results[name][0]:
            if isinstance(all_results[name][0][key], (int, float, np.integer, np.floating)):
                avg_results[name][key] = np.mean([r[key] for r in all_results[name]])
            else:
                avg_results[name][key] = deepcopy(all_results[name][0][key])

        # Compute averaged classification report
        stacked_preds = np.vstack(all_preds[name])
        averaged_preds = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x.astype(int))),
            axis=0,
            arr=stacked_preds
        )
        print(f"--- Averaged Classification Report for {name} ---")
        print(classification_report(y_test, averaged_preds))
        print("\n")

    return avg_results