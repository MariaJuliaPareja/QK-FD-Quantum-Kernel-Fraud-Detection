import matplotlib.pyplot as plt
import pennylane as qml

from preprocess import prepare_qsvm_data

N_QUBITS = 4
N_REPEATS = 2

dev = qml.device("default.qubit", wires=N_QUBITS)


def feature_map(x, wires):
    """ZZ-style feature map (Havlicek et al. 2019): Hadamards + data-encoded RZ
    rotations + pairwise ZZ entangling gates, repeated N_REPEATS times.
    qml.IQPEmbedding implements exactly this construction.
    """
    qml.IQPEmbedding(x, wires=wires, n_repeats=N_REPEATS)


@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1, wires=range(N_QUBITS))
    qml.adjoint(feature_map)(x2, wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))


def quantum_kernel(x1, x2):
    return kernel_circuit(x1, x2)[0]


def compute_training_kernel_matrix(X_train, out_path="kernel_matrix.png"):
    print(f"\nBuilding {N_QUBITS}-qubit ZZ-style feature map kernel...")
    K_train = qml.kernels.kernel_matrix(X_train, X_train, quantum_kernel)
    print("Training kernel matrix shape:", K_train.shape)

    plt.figure(figsize=(6, 5))
    plt.imshow(K_train, cmap="viridis")
    plt.colorbar(label="Kernel value")
    plt.title("Quantum Kernel Matrix (Training Set)")
    plt.xlabel("Sample index")
    plt.ylabel("Sample index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved heatmap to {out_path}")

    return K_train


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, top_features, scaler = prepare_qsvm_data()
    compute_training_kernel_matrix(X_train)
