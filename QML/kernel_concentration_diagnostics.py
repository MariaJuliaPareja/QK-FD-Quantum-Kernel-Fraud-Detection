import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from preprocess import prepare_qsvm_data_raw, scale_features

N_QUBITS = 4
dev = qml.device("default.qubit", wires=N_QUBITS)


def iqp_feature_map(x, wires, n_repeats):
    qml.IQPEmbedding(x, wires=wires, n_repeats=n_repeats)


def angle_feature_map(x, wires):
    qml.AngleEmbedding(x, wires=wires, rotation="Y")


def make_kernel(feature_map, **fm_kwargs):
    @qml.qnode(dev)
    def circuit(x1, x2):
        feature_map(x1, wires=range(N_QUBITS), **fm_kwargs)
        qml.adjoint(feature_map)(x2, wires=range(N_QUBITS), **fm_kwargs)
        return qml.probs(wires=range(N_QUBITS))

    def kernel(x1, x2):
        return circuit(x1, x2)[0]

    return kernel


def off_diagonal_stats(K):
    mask = ~np.eye(K.shape[0], dtype=bool)
    off_diag = K[mask]
    return off_diag.mean(), off_diag.std()


def run_variant(name, X, kernel_fn):
    print(f"\nComputing kernel matrix: {name}...")
    K = qml.kernels.kernel_matrix(X, X, kernel_fn)
    mean_off, std_off = off_diagonal_stats(K)
    print(f"  shape={K.shape}, off-diagonal mean={mean_off:.4f}, std={std_off:.4f}")
    return K, mean_off, std_off


def main():
    X_train_raw, X_test_raw, y_train, y_test, top_features = prepare_qsvm_data_raw()

    X_2pi, _, _ = scale_features(X_train_raw, X_test_raw, feature_range=(0, 2 * np.pi))
    X_pi, _, _ = scale_features(X_train_raw, X_test_raw, feature_range=(0, np.pi))

    iqp_reps2 = make_kernel(iqp_feature_map, n_repeats=2)
    iqp_reps1 = make_kernel(iqp_feature_map, n_repeats=1)
    angle = make_kernel(angle_feature_map)

    variants = [
        ("Baseline: IQP reps=2, [0,2π]", X_2pi, iqp_reps2),
        ("Var 1: IQP reps=1, [0,2π]", X_2pi, iqp_reps1),
        ("Var 2: IQP reps=2, [0,π]", X_pi, iqp_reps2),
        ("Var 3: AngleEmbedding, [0,2π]", X_2pi, angle),
    ]

    results = []
    for name, X, kernel_fn in variants:
        K, mean_off, std_off = run_variant(name, X, kernel_fn)
        results.append((name, K, mean_off, std_off))

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    im = None
    for ax, (title, K, mean_off, std_off) in zip(axes.flat, results):
        im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{title}\noff-diag mean={mean_off:.3f} ± {std_off:.3f}", fontsize=10)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Sample index")
    fig.colorbar(im, ax=axes, label="Kernel value", shrink=0.8)

    out_path = "kernel_concentration_comparison.png"
    plt.savefig(out_path, dpi=200)
    print(f"\nSaved comparison heatmap to {out_path}")

    print("\nSummary (off-diagonal mean ± std) — higher mean = less concentration:")
    for title, _, mean_off, std_off in results:
        print(f"  {title}: {mean_off:.4f} ± {std_off:.4f}")


if __name__ == "__main__":
    main()
