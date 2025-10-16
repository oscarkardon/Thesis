import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_synthetic_vs_real(X_real, y_real, X_synth, y_synth, protected_attrs=None, max_features=5):
    """
    Compare real vs synthetic datasets with multiple visual and statistical checks.

    Args:
        X_real, y_real: pd.DataFrame, pd.Series - real data
        X_synth, y_synth: pd.DataFrame, pd.Series - synthetic data
        protected_attrs: list[str] - optional list of categorical columns to compare balance
        max_features: int - how many features to visualize
    """
    # --- 0. Combine ---
    X_real_copy = X_real.copy()
    X_real_copy["source"] = "real"
    X_real_copy["label"] = y_real.values

    X_synth_copy = X_synth.copy()
    X_synth_copy["source"] = "synthetic"
    X_synth_copy["label"] = y_synth.values

    combined = pd.concat([X_real_copy, X_synth_copy], axis=0, ignore_index=True)

    # Separate numeric vs categorical
    # Exclude 'label' when selecting numeric columns
    numeric_cols = combined.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['label', 'sex', 'race']] # Exclude label, sex, and race

    # Select categorical columns, exclude source and label
    categorical_cols = combined.select_dtypes(exclude=np.number).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['source', 'label']]


    # --- 1. Summary statistics ---
    print("\n📊 Summary statistics (numeric features only):")
    display(combined[["source"] + numeric_cols].groupby("source").describe().T)

    # --- 2. Distribution plots (numeric) ---
    for col in numeric_cols[:max_features]:
        plt.figure(figsize=(6,4))
        sns.kdeplot(data=combined, x=col, hue="source", common_norm=False, fill=True, alpha=0.4)
        plt.title(f"Distribution of {col} (Real vs Synthetic)")
        plt.show()

    # --- 3. Distribution plots (categorical) ---
    for col in categorical_cols[:max_features]:
        plt.figure(figsize=(6,4))
        sns.countplot(data=combined, x=col, hue="source")
        plt.title(f"Category distribution of {col} (Real vs Synthetic)")
        plt.xticks(rotation=45)
        plt.show()

    # --- 4. Correlation comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    sns.heatmap(combined[combined['source'] == 'real'][numeric_cols].corr(), ax=axes[0], cmap="coolwarm", center=0)
    axes[0].set_title("Real Data Correlations")
    sns.heatmap(combined[combined['source'] == 'synthetic'][numeric_cols].corr(), ax=axes[1], cmap="coolwarm", center=0)
    axes[1].set_title("Synthetic Data Correlations")
    plt.show()

    # --- 5. PCA Visualization ---
    pca = PCA(n_components=2)
    # Ensure only numeric columns are used for PCA, handle potential NaNs
    X_embedded = pca.fit_transform(combined[numeric_cols].fillna(0))
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1],
                    hue=combined["source"], style=combined["label"], alpha=0.6)
    plt.title("PCA: Real vs Synthetic")
    plt.show()

    # --- 6. t-SNE Visualization ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    # Ensure only numeric columns are used for t-SNE, handle potential NaNs
    X_embedded_tsne = tsne.fit_transform(combined[numeric_cols].fillna(0))
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=X_embedded_tsne[:,0], y=X_embedded_tsne[:,1],
                    hue=combined["source"], style=combined["label"], alpha=0.6)
    plt.title("t-SNE: Real vs Synthetic")
    plt.show()

    # --- 7. Group counts by label ---
    print("\n📌 Group counts by label and source:")
    print(combined.groupby(["source", "label"]).size())

    # --- 8. Protected attribute balance (optional) ---
    if protected_attrs:
        for attr in protected_attrs:
            if attr in combined.columns:
                print(f"\n⚖️ Balance check for {attr}:")
                print(combined.groupby(["source", attr, "label"]).size())


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def quadrant_analysis_binary_income(X_real, X_synth, numeric_cols,
                                   label_real, label_synth,
                                   sex_col='sex', income_col='income'):
    """
    Split data into 4 quadrants based on sex (0=female, 1=male) and binary income (0=low, 1=high):
    - Q1: High income men
    - Q2: Low income men
    - Q3: High income women
    - Q4: Low income women

    Then compute correlation distance, heatmaps, KDEs, and summary stats.
    """

    # --- Combine X and label into one dataframe ---
    real_df = X_real.copy()
    real_df['source'] = 'real'
    real_df[income_col] = label_real.values

    synth_df = X_synth.copy()
    synth_df['source'] = 'synthetic'
    synth_df[income_col] = label_synth.values

    combined = pd.concat([real_df, synth_df], ignore_index=True)

    # --- Assign quadrants using binary sex and income ---
    def assign_quadrant(row):
        sex = row[sex_col]
        income = row[income_col]
        if sex == 1 and income == 1:
            return 'Q1: High income men'
        elif sex == 1 and income == 0:
            return 'Q2: Low income men'
        elif sex == 0 and income == 1:
            return 'Q3: High income women'
        elif sex == 0 and income == 0:
            return 'Q4: Low income women'
        else:
            return 'Other'

    combined['quadrant'] = combined.apply(assign_quadrant, axis=1)

    # --- Print counts per quadrant ---
    group_counts = combined.groupby(['quadrant', 'source']).size().to_dict()
    print("ℹ️ Counts per quadrant and source:", group_counts)
    max_count = max(group_counts.values())
    print("Max count across quadrants:", max_count)

    # --- Analysis per quadrant ---
    quadrants = ['Q1: High income men', 'Q2: Low income men',
                 'Q3: High income women', 'Q4: Low income women']

    for q in quadrants:
        real_q = combined[(combined['source']=='real') & (combined['quadrant']==q)][numeric_cols]
        synth_q = combined[(combined['source']=='synthetic') & (combined['quadrant']==q)][numeric_cols]

        if real_q.empty or synth_q.empty:
            print(f"\n⚠️ Skipping {q} (missing data)")
            continue

        print(f"\n===== {q} =====")

        # Correlation distance
        corr_real = real_q.corr().fillna(0)
        corr_synth = synth_q.corr().fillna(0)

        # Align columns in case of slight mismatch
        corr_real, corr_synth = corr_real.align(corr_synth, join='inner', axis=0)
        corr_real, corr_synth = corr_real.align(corr_synth, join='inner', axis=1)

        # Frobenius norm of difference
        pcd = np.sqrt(np.nansum((corr_real.values - corr_synth.values)**2))
        print(f"Correlation distance (pCD) = {pcd:.4f}")

        # Heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        sns.heatmap(corr_real, cmap='viridis', center=0, ax=axes[0])
        axes[0].set_title(f"Real {q}")
        sns.heatmap(corr_synth, cmap='viridis', center=0, ax=axes[1])
        axes[1].set_title(f"Synthetic {q}")
        plt.show()

        # KDE plots (limit to first few for clarity)
        for col in numeric_cols[:5]:
            plt.figure(figsize=(6,4))
            sns.kdeplot(real_q[col], label='Real', fill=True, alpha=0.4)
            sns.kdeplot(synth_q[col], label='Synthetic', fill=True, alpha=0.4)
            plt.title(f"{col} Distribution in {q}")
            plt.legend()
            plt.show()

        # Summary stats
        print(f"Summary stats for {q} (Real):")
        display(real_q.describe())
        print(f"Summary stats for {q} (Synthetic):")
        display(synth_q.describe())
