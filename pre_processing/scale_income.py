import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#This new function allows me to use the same code passing in the scaler object when approptiate
def preprocess_data(X, y, scaler=None):
    """
    Preprocess features and labels for fairness experiments.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    y : pd.Series or pd.DataFrame
        Target labels.
    scaler : MinMaxScaler or None
        If None, fit a new scaler on numeric columns of X.
        If provided, use it to transform X without refitting.

    Returns
    -------
    X_clean : pd.DataFrame
        Processed features (categoricals encoded, numerics scaled).
    y_clean : pd.Series
        Binary target (0/1).
    scaler : MinMaxScaler
        Fitted scaler object.
    """

    X_clean = X.copy()

    # ---- Normalize column strings (avoid trailing spaces / caps) ----
    if 'race' in X_clean.columns and X_clean['race'].dtype == 'object':
        X_clean['race'] = X_clean['race'].str.strip().str.lower()
    if 'sex' in X_clean.columns and X_clean['sex'].dtype == 'object':
        X_clean['sex'] = X_clean['sex'].str.strip().str.capitalize()


    # ---- Encode categorical/protected variables ----
    if 'sex' in X_clean.columns:
        X_clean['sex'] = np.where(X_clean['sex'] == 'Male', 1, 0)
    if 'race' in X_clean.columns:
        X_clean['race'] = np.where(X_clean['race'] == 'white', 1, 0)

    # ---- Clean target labels ----
    y = y.squeeze()  # ensures Series
    if y.dtype == object or y.dtype == str:
        y_clean = y.replace({'<=50K.': '<=50K', '>50K.': '>50K'})
        y_clean = np.where(y_clean == '<=50K', 0, 1)
    else:
        y_clean = y.astype(int)

    # ---- Age bucketing ----
    if 'age' in X_clean.columns:
        bins = [0, 10, 20, 30, 40, 50, 60, 70, np.inf]
        labels = [0, 10, 20, 30, 40, 50, 60, 70]
        X_clean['age'] = pd.cut(X_clean['age'], bins=bins, labels=labels, right=False).astype(int)

    # ---- Scale numeric features (exclude sex, race) ----
    numeric_cols = X_clean.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['sex', 'race']]

    if scaler is None:
      scaler = MinMaxScaler()
      X_clean[numeric_cols] = scaler.fit_transform(X_clean[numeric_cols])
    else:
        X_clean[numeric_cols] = scaler.transform(X_clean[numeric_cols])

    # Ensure index is preserved
    X_clean = pd.DataFrame(X_clean, columns=X_clean.columns, index=X.index)

    return X_clean, pd.Series(y_clean, name='income'), scaler

def normalize_columns(X, y):
    """Normalize sex, race, and income columns for consistency."""
    X = X.copy()
    y = y.copy()

    if "sex" in X.columns and X["sex"].dtype == "object":
        X["sex"] = X["sex"].str.strip().str.capitalize()
        X["sex"] = X["sex"].map({"Male": 1, "Female": 0})
    elif "sex" in X.columns:
        X["sex"] = X["sex"].astype(int)

    if "race" in X.columns and X["race"].dtype == "object":
        X["race"] = X["race"].str.strip().str.lower()
        X["race"] = (X["race"] == "white").astype(int)
    elif "race" in X.columns:
        X["race"] = X["race"].astype(int)

    # Normalize target labels
    y = y.replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    y = (y != '<=50K').astype(int)

    return X, y