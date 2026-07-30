import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data_loader import load_creditcard_data

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
N_FEATURES = 4
N_PER_CLASS = 100
RANDOM_STATE = 42


def select_top_variance_features(df, columns=V_COLUMNS, n=N_FEATURES):
    variances = df[columns].var().sort_values(ascending=False)
    return variances.index[:n].tolist(), variances


def build_balanced_subset(df, n_per_class=N_PER_CLASS, random_state=RANDOM_STATE):
    fraud = df[df["Class"] == 1].sample(n=n_per_class, random_state=random_state)
    legit = df[df["Class"] == 0].sample(n=n_per_class, random_state=random_state)
    subset = pd.concat([fraud, legit]).sample(frac=1, random_state=random_state)
    return subset.reset_index(drop=True)


def prepare_qsvm_data_raw(test_size=0.2, random_state=RANDOM_STATE):
    df = load_creditcard_data()

    top_features, variances = select_top_variance_features(df)
    print(f"Top {N_FEATURES} features by variance (of V1-V28): {top_features}")
    print(variances.loc[top_features].to_string())

    subset = build_balanced_subset(df, N_PER_CLASS, random_state)
    print("\nBalanced subset shape:", subset.shape)
    print("Class balance in subset:")
    print(subset["Class"].value_counts().sort_index().to_string())

    X = subset[top_features].values
    y = subset["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, top_features


def scale_features(X_train, X_test, feature_range=(0, 2 * np.pi)):
    scaler = MinMaxScaler(feature_range=feature_range)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_qsvm_data(test_size=0.2, random_state=RANDOM_STATE, feature_range=(0, 2 * np.pi)):
    X_train, X_test, y_train, y_test, top_features = prepare_qsvm_data_raw(test_size, random_state)

    X_train, X_test, scaler = scale_features(X_train, X_test, feature_range)

    print("\nX_train shape:", X_train.shape, "| X_test shape:", X_test.shape)
    print("Train class balance:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Test class balance:", dict(zip(*np.unique(y_test, return_counts=True))))

    return X_train, X_test, y_train, y_test, top_features, scaler


if __name__ == "__main__":
    prepare_qsvm_data()
