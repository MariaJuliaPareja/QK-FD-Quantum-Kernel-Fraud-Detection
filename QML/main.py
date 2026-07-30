import qiskit
import qiskit_machine_learning
import qiskit_aer
import sklearn
import pandas
import matplotlib

from data_loader import load_creditcard_data, summarize
from preprocess import prepare_qsvm_data
from quantum_kernel import compute_training_kernel_matrix

print("qiskit:", qiskit.__version__)
print("qiskit-machine-learning:", qiskit_machine_learning.__version__)
print("qiskit-aer:", qiskit_aer.__version__)
print("scikit-learn:", sklearn.__version__)
print("pandas:", pandas.__version__)
print("matplotlib:", matplotlib.__version__)

print()
print("Loading creditcard fraud dataset...")
df = load_creditcard_data()
summarize(df)

print()
print("Preparing balanced subset for QSVM demo...")
X_train, X_test, y_train, y_test, top_features, scaler = prepare_qsvm_data()

print()
print("Computing quantum kernel matrix (this takes a few minutes)...")
K_train = compute_training_kernel_matrix(X_train)
