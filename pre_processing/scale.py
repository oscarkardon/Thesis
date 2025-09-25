import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(X, y):
    X_clean = X.copy()

    # ---- Encode categorical/protected variables ----
    X_clean['sex'] = np.where(X_clean['sex'] == 'Male', 1, 0)
    X_clean['race'] = np.where(X_clean['race'].str.strip().str.lower() == 'white', 1, 0)

    # ---- Clean target labels ----
    y_clean = y.iloc[:, 0].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    y_clean = np.where(y_clean == '<=50K', 0, 1)

    # ---- Age bucketing ----
    bins = [0, 10, 20, 30, 40, 50, 60, 70, np.inf]
    labels = [0, 10, 20, 30, 40, 50, 60, 70]
    X_clean['age'] = pd.cut(X_clean['age'], bins=bins, labels=labels, right=False).astype(int)

    # ---- Scale only numeric features (exclude sex and race) ----
    numeric_cols = X_clean.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['sex', 'race']]

    scaler = MinMaxScaler()
    X_clean[numeric_cols] = scaler.fit_transform(X_clean[numeric_cols])

    return X_clean, pd.Series(y_clean, name='income')
