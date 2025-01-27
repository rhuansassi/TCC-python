import numpy as np


def process_output_nn(XTrainStruct, YTrainStruct, XValidStruct, YValidStruct, onlyTrainData):


    NInputsTrain, numSymbolsTrain = XTrainStruct.shape
    NInputsValid, numSymbolsValid = XValidStruct.shape

    # Alocar saída complexa
    XTrainRestruct = np.zeros((NInputsTrain // 2, numSymbolsTrain), dtype=complex)
    YTrainRestruct = np.zeros((NInputsTrain // 2, numSymbolsTrain), dtype=complex)
    XValidRestruct = np.zeros((NInputsValid // 2, numSymbolsValid), dtype=complex)
    YValidRestruct = np.zeros((NInputsValid // 2, numSymbolsValid), dtype=complex)

    # Índices para real e imag (0-based em Python)
    realIdxTrain = np.arange(0, NInputsTrain, 2)
    imagIdxTrain = np.arange(1, NInputsTrain, 2)

    # Reconstrução para dados de treinamento
    # Podemos fazer por toda a matriz de uma vez usando fancy indexing:
    XTrainRestruct = XTrainStruct[realIdxTrain, :] + 1j * XTrainStruct[imagIdxTrain, :]
    YTrainRestruct = YTrainStruct[realIdxTrain, :] + 1j * YTrainStruct[imagIdxTrain, :]

    if not onlyTrainData:
        realIdxValid = np.arange(0, NInputsValid, 2)
        imagIdxValid = np.arange(1, NInputsValid, 2)

        XValidRestruct = XValidStruct[realIdxValid, :] + 1j * XValidStruct[imagIdxValid, :]
        YValidRestruct = YValidStruct[realIdxValid, :] + 1j * YValidStruct[imagIdxValid, :]

    return XTrainRestruct, YTrainRestruct, XValidRestruct, YValidRestruct
