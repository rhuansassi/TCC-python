import os
from tabnanny import verbose

import numpy as np
import tensorflow as tf

from neuralnetworks.rvnn import predict
from scripts.process_input_nn import process_input_nn
from scripts.process_output_nn import process_output_nn

# Forçar uso da CPU (opcional, caso queira garantir que não use GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def perform_inference_networks(X,
                               netSCF, scalingFactorSCF,
                               trainedNetRVNN, scalingFactorRVNN):
    # Se netSCF.y é complexo, outSCF e outRVNN são complexos
    outSCF = np.zeros_like(X, dtype=complex)
    outRVNN = np.zeros_like(X, dtype=complex)

    numSymbols = X.shape[1]
    for symbol in range(numSymbols):
        x = X[:, symbol]

        # Inferência SCFNN
        netSCF.y = netSCF.feedforward(x / scalingFactorSCF)
        outSCF[:, symbol] = netSCF.y * scalingFactorSCF

        # Ajustar x para RVNN (de (N,) -> (N,1))
        x_2d = (x / scalingFactorRVNN)[:, None]

        # Substituir 0 por arrays vazias 2D conforme a lógica do código original
        XValid_empty = np.empty((0, 0))
        YValid_empty = np.empty((0, 0))

        # Processar input da RVNN
        rvnnInput, _, _, _ = process_input_nn(x_2d, X, XValid_empty, YValid_empty, True)

        # Previsão RVNN na CPU
        with tf.device('/CPU:0'):
            predicted = predict(trainedNetRVNN, rvnnInput)

        scaled_predicted = predicted.T * scalingFactorRVNN

        outRVNN_col, _, _, _ = process_output_nn(scaled_predicted, predicted.T, predicted.T, predicted.T, True)

        # Remover dimensão extra se houver
        outRVNN[:, symbol] = outRVNN_col.squeeze()

    return outSCF, outRVNN
