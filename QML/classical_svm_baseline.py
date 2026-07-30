from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from data_loader import load_creditcard_data
from preprocess import N_PER_CLASS, RANDOM_STATE, V_COLUMNS, build_balanced_subset

CANDIDATE_FEATURES = V_COLUMNS[:4]  # same top-variance features used for the QSVM (V1-V4)
OUT_PATH = "classical_svm_fraud_boundary.png"


def select_clearest_separating_pair(subset, candidates=CANDIDATE_FEATURES):
    """Pick the 2 features with the clearest class separation via cross-validated
    RBF-SVM accuracy, rather than assuming the top-2-by-variance pair (V1, V2) is best.
    """
    y = subset["Class"].values
    scores = {}
    for a, b in combinations(candidates, 2):
        X = subset[[a, b]].values
        scores[(a, b)] = cross_val_score(SVC(kernel="rbf"), X, y, cv=5).mean()

    print("Cross-validated RBF-SVM accuracy per feature pair:")
    for pair, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {pair}: {score:.4f}")

    best_pair = max(scores, key=scores.get)
    return best_pair


def plot_decision_boundary(X, y, feature_names, out_path=OUT_PATH):
    clf = SVC(kernel="rbf")
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

    plt.scatter(
        X[y == 0, 0], X[y == 0, 1],
        c="tab:blue", label="Non-fraud", edgecolor="k", s=40,
    )
    plt.scatter(
        X[y == 1, 0], X[y == 1, 1],
        c="tab:orange", label="Fraud", edgecolor="k", s=40,
    )

    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Classical SVM (RBF) — Credit Card Fraud, 2 PCA Components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved decision boundary plot to {out_path}")

    return clf


def main():
    df = load_creditcard_data()
    subset = build_balanced_subset(df, N_PER_CLASS, RANDOM_STATE)

    best_pair = select_clearest_separating_pair(subset)
    print(f"\nUsing feature pair: {best_pair}")

    X = subset[list(best_pair)].values
    y = subset["Class"].values

    plot_decision_boundary(X, y, best_pair)


if __name__ == "__main__":
    main()
