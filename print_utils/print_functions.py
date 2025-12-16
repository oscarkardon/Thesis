import numpy as np
import pandas as pd
from IPython.display import display

def save_results_to_csv(all_results_pre, all_results_post, model_names, pcd_unnorm, pcd_norm, filename="model_results.csv"):
    """
    Save model results to CSV, including PCD scores for the Post-LLM phase.
    """
    rows = []

    # Process individual models
    for model, pre, post in zip(model_names, all_results_pre, all_results_post):
        # Clean dictionaries (avoiding side effects on original dicts)
        pre_clean = {k: v for k, v in pre.items() if k not in ['classification_report', 'y_pred']}
        post_clean = {k: v for k, v in post.items() if k not in ['classification_report', 'y_pred']}
        
        # Add Pre-LLM row (PCD is NaN because it's the baseline)
        rows.append({
            "Model": model,
            "Iteration": "Pre-LLM",
            "pcd_unnormalized": np.nan,
            "pcd_normalized": np.nan,
            **pre_clean
        })

        # Add Post-LLM row (Include PCD scores here)
        rows.append({
            "Model": model,
            "Iteration": "Post-LLM",
            "pcd_unnormalized": pcd_unnorm,
            "pcd_normalized": pcd_norm,
            **post_clean
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

    # Reorder columns to put PCD near the front for visibility
    cols = ['Model', 'Iteration', 'pcd_normalized', 'accuracy', 'tpr_difference'] # Adjust as needed
    remaining_cols = [c for c in df.columns if c not in cols]
    df = df[cols + remaining_cols]

    df.to_csv(filename, index=False)
    print(f"Results (including PCD scores) saved to {filename}")
    display(df)