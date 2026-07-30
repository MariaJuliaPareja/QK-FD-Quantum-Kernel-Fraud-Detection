import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.svm import SVC

from preprocess import prepare_qsvm_data

N_QUBITS = 4
dev = qml.device("default.qubit", wires=N_QUBITS)


def angle_feature_map(x, wires):
    qml.AngleEmbedding(x, wires=wires, rotation="Y")


@qml.qnode(dev)
def kernel_circuit(x1, x2):
    angle_feature_map(x1, wires=range(N_QUBITS))
    qml.adjoint(angle_feature_map)(x2, wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))


def quantum_kernel(x1, x2):
    return kernel_circuit(x1, x2)[0]


def evaluate(y_test, y_pred):
    return (
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        recall_score(y_test, y_pred),
    )


def main():
    X_train, X_test, y_train, y_test, top_features, scaler = prepare_qsvm_data()

    classical_clf = SVC(kernel="rbf")
    classical_clf.fit(X_train, y_train)
    classical_metrics = evaluate(y_test, classical_clf.predict(X_test))

    print("Computing quantum kernel matrices (AngleEmbedding)...")
    K_train = qml.kernels.kernel_matrix(X_train, X_train, quantum_kernel)
    K_test = qml.kernels.kernel_matrix(X_test, X_train, quantum_kernel)

    quantum_clf = SVC(kernel="precomputed")
    quantum_clf.fit(K_train, y_train)
    quantum_metrics = evaluate(y_test, quantum_clf.predict(K_test))

    print(
        f"Classical RBF -> accuracy={classical_metrics[0]:.4f}, "
        f"f1={classical_metrics[1]:.4f}, recall={classical_metrics[2]:.4f}"
    )
    print(
        f"Quantum QSVM  -> accuracy={quantum_metrics[0]:.4f}, "
        f"f1={quantum_metrics[1]:.4f}, recall={quantum_metrics[2]:.4f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    metric_names = ["Accuracy", "F1", "Recall"]
    x = np.arange(len(metric_names))
    width = 0.35

    ax1 = axes[0]
    ax1.bar(x - width / 2, classical_metrics, width, label="Classical RBF", color="tab:blue")
    ax1.bar(x + width / 2, quantum_metrics, width, label="Quantum QSVM", color="tab:orange")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Score")
    ax1.set_title("Classical RBF vs Quantum AngleEmbedding QSVM")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    im = ax2.imshow(K_train, cmap="viridis", vmin=0, vmax=1)
    ax2.set_title("Quantum Kernel Matrix (AngleEmbedding, Training Set)")
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Sample index")
    fig.colorbar(im, ax=ax2, label="Kernel value", fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = "results.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
