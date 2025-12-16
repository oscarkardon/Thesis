import pandas as pd
from IPython.display import display

def print_averaged_results(results, model_name):
    print(f"\n📊 Averaged Results for {model_name}")
    print("=" * 40)
    for k, v in results.items():
        val = f"{v:.4f}" if isinstance(v, float) or isinstance(v, np.floating) else str(v)
        print(f"{k.replace('_', ' ').title():<25}: {val}")


def save_results_to_csv(all_results_pre, all_results_post, model_names, filename="model_results.csv"):
    """
    Save model results to CSV, adding average rows for Pre-LLM, Post-LLM, and a difference row.

    Parameters:
        all_results_pre (list of dict): list of pre-LLM results per model
        all_results_post (list of dict): list of post-LLM results per model
        model_names (list of str): model names corresponding to results
        filename (str): CSV file name
    """
    rows = []

    # Add model rows
    for model, pre, post in zip(model_names, all_results_pre, all_results_post):
        # Remove classification report and y_pred from dictionary to save to csv
        pre.pop('classification_report', None)
        pre.pop('y_pred', None)
        post.pop('classification_report', None)
        post.pop('y_pred', None)
        
        # Add a row for the pre-LLM model
        rows.append({
            "Model": model,
            "Iteration": "Pre-LLM",
            **pre
        })

        # Add a row for the post-LLM model
        rows.append({
            "Model": model,
            "Iteration": "Post-LLM",
            **post
        })

    df = pd.DataFrame(rows)

    # Compute averages for numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    avg_pre = df[df["Iteration"] == "Pre-LLM"][numeric_cols].mean()
    avg_post = df[df["Iteration"] == "Post-LLM"][numeric_cols].mean()

    # Compute difference row (Post - Pre)
    diff = avg_post - avg_pre

    # Add average rows and difference row
    df = pd.concat([
        df,
        pd.DataFrame([{"Model": "Average", "Iteration": "Pre-LLM", **avg_pre}]),
        pd.DataFrame([{"Model": "Average", "Iteration": "Post-LLM", **avg_post}]),
        pd.DataFrame([{"Model": "Difference", "Iteration": "Post-LLM - Pre-LLM", **diff}])
    ], ignore_index=True)

    df.to_csv(filename, index=False)
    print(f"Results (including averages and difference) saved to {filename}")
    display(df)