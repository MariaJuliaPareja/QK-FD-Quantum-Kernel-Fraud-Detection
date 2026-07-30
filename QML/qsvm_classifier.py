import pennylane as qml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.svm import SVC

from preprocess import prepare_qsvm_data

N_QUBITS = 4
dev = qml.device("default.qubit", wires=N_QUBITS)


def angle_feature_map(x, wires):
    """Var 3 solution from the concentration diagnostics: plain AngleEmbedding,
    no entanglement, [0, 2*pi] range — the variant with the healthiest kernel matrix.
    """
    qml.AngleEmbedding(x, wires=wires, rotation="Y")


@qml.qnode(dev)
def kernel_circuit(x1, x2):
    angle_feature_map(x1, wires=range(N_QUBITS))
    qml.adjoint(angle_feature_map)(x2, wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))


def quantum_kernel(x1, x2):
    return kernel_circuit(x1, x2)[0]


def main():
    X_train, X_test, y_train, y_test, top_features, scaler = prepare_qsvm_data()

    print("\nComputing train kernel matrix (AngleEmbedding, no entanglement)...")
    K_train = qml.kernels.kernel_matrix(X_train, X_train, quantum_kernel)
    print("K_train shape:", K_train.shape)

    print("Computing test-vs-train kernel matrix...")
    K_test = qml.kernels.kernel_matrix(X_test, X_train, quantum_kernel)
    print("K_test shape:", K_test.shape)

    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)

    y_pred = clf.predict(K_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- QSVM (AngleEmbedding kernel) test performance ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"Recall (fraud detection): {recall:.4f}")
    print("Confusion matrix:")
    print("              pred_legit  pred_fraud")
    print(f"true_legit   {cm[0, 0]:>10d}  {cm[0, 1]:>10d}")
    print(f"true_fraud   {cm[1, 0]:>10d}  {cm[1, 1]:>10d}")

    return acc, f1, recall, cm


if __name__ == "__main__":
    main()
