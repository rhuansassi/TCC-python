import numpy as np


def process_input_nn(XTrain, YTrain, XValid, YValid, onlyTrainData):
    NInputsTrain, numSymbolsTrain = XTrain.shape
    NInputsValid, numSymbolsValid = XValid.shape

    XTrainStruct = np.zeros((2 * NInputsTrain, numSymbolsTrain))
    YTrainStruct = np.zeros((2 * NInputsTrain, numSymbolsTrain))
    XValidStruct = np.zeros((2 * NInputsValid, numSymbolsValid))
    YValidStruct = np.zeros((2 * NInputsValid, numSymbolsValid))

    realIdxTrain = np.arange(0, 2 * NInputsTrain, 2)
    imagIdxTrain = np.arange(1, 2 * NInputsTrain, 2)

    for symbol in range(numSymbolsTrain):
        XTrainStruct[realIdxTrain, symbol] = np.real(XTrain[:, symbol])
        XTrainStruct[imagIdxTrain, symbol] = np.imag(XTrain[:, symbol])
        YTrainStruct[realIdxTrain, symbol] = np.real(YTrain[:, symbol])
        YTrainStruct[imagIdxTrain, symbol] = np.imag(YTrain[:, symbol])

    if not onlyTrainData:
        realIdxValid = np.arange(0, 2 * NInputsValid, 2)
        imagIdxValid = np.arange(1, 2 * NInputsValid, 2)

        for symbol in range(numSymbolsValid):
            XValidStruct[realIdxValid, symbol] = np.real(XValid[:, symbol])
            XValidStruct[imagIdxValid, symbol] = np.imag(XValid[:, symbol])
            YValidStruct[realIdxValid, symbol] = np.real(YValid[:, symbol])
            YValidStruct[imagIdxValid, symbol] = np.imag(YValid[:, symbol])

    return XTrainStruct, YTrainStruct, XValidStruct, YValidStruct
