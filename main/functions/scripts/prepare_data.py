# prepare_data.py

import numpy as np

def prepare_data_for_scffnn(X):
    """
    Prepara os dados de entrada para a SCFFNN.
    """
    X_real = np.real(X.T)
    X_imag = np.imag(X.T)
    return np.concatenate((X_real, X_imag), axis=1)

def prepare_labels_for_scffnn(Y):
    """
    Prepara os dados de saída (rótulos) para a SCFFNN.
    """
    Y_real = np.real(Y.T)
    Y_imag = np.imag(Y.T)
    return np.concatenate((Y_real, Y_imag), axis=1)

def reconstruct_complex_output(Y_pred):
    """
    Reconstrói a saída complexa a partir das partes real e imaginária.
    """
    n = Y_pred.shape[1] // 2
    Y_real = Y_pred[:, :n]
    Y_imag = Y_pred[:, n:]
    return Y_real + 1j * Y_imag
