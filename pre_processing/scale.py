import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(X, y):
    X_clean = X.copy()

    # Encode categorical variables
    X_clean['sex'] = np.where(X['sex'] == 'Male', 1, 0)
    X_clean['race'] = np.where(X['race'].str.strip() == 'White', 1, 0)

    # Clean target labels
    y_clean = y.iloc[:, 0].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    y_clean = y_clean.rename('income')
    y_clean = np.where(y_clean == '<=50K', 0, 1)

    # Age bucketing
    X_clean['age'] = np.where(X['age'] >= 70, 70, X['age'])
    X_clean['age'] = np.where((X['age'] >= 60) & (X['age'] < 70), 60, X['age'])
    X_clean['age'] = np.where((X['age'] >= 50) & (X['age'] < 60), 50, X['age'])
    X_clean['age'] = np.where((X['age'] >= 40) & (X['age'] < 50), 40, X['age'])
    X_clean['age'] = np.where((X['age'] >= 30) & (X['age'] < 40), 30, X['age'])
    X_clean['age'] = np.where((X['age'] >= 20) & (X['age'] < 30), 20, X['age'])
    X_clean['age'] = np.where((X['age'] >= 10) & (X['age'] < 20), 10, X['age'])
    X_clean['age'] = np.where(X['age'] < 10, 0, X['age'])

    # Scale numeric features
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns)

    return X_scaled, y_clean
