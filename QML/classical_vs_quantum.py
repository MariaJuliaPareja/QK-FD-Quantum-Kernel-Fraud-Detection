from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.svm import SVC

from preprocess import prepare_qsvm_data

# Quantum results from qsvm_classifier.py (AngleEmbedding kernel), same train/test split
QUANTUM_RESULTS = {
    "accuracy": 0.8750,
    "f1": 0.8718,
    "recall": 0.8500,
    "cm": [[18, 2], [3, 17]],
}


def main():
    # Same 4 features (V1-V4), same balanced subset, same 80/20 stratified split,
    # same [0, 2*pi] scaling as used for the QSVM in qsvm_classifier.py.
    X_train, X_test, y_train, y_test, top_features, scaler = prepare_qsvm_data()

    clf = SVC(kernel="rbf")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Classical SVM (RBF kernel) test performance ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"Recall (fraud detection): {recall:.4f}")
    print("Confusion matrix:")
    print("              pred_legit  pred_fraud")
    print(f"true_legit   {cm[0, 0]:>10d}  {cm[0, 1]:>10d}")
    print(f"true_fraud   {cm[1, 0]:>10d}  {cm[1, 1]:>10d}")

    print("\n--- Classical RBF vs Quantum AngleEmbedding QSVM ---")
    print(f"{'Metric':<12} {'Classical RBF':>15} {'Quantum QSVM':>15}")
    print(f"{'Accuracy':<12} {acc:>15.4f} {QUANTUM_RESULTS['accuracy']:>15.4f}")
    print(f"{'F1':<12} {f1:>15.4f} {QUANTUM_RESULTS['f1']:>15.4f}")
    print(f"{'Recall':<12} {recall:>15.4f} {QUANTUM_RESULTS['recall']:>15.4f}")

    return acc, f1, recall, cm


if __name__ == "__main__":
    main()
