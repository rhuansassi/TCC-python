import numpy as np

from scripts.utils import do_qamdemodulate

"""
def get_SER_Comparison(p, pCE, txLabels, XLabels, rxSymbols_LS, rxSymbols_LMMSE, scfOutput, rvNNOuput):


    # Número de símbolos OFDM (colunas)
    num_symbols = rxSymbols_LS.shape[1]

    # Vetores para acumular as contagens de erros por símbolo OFDM
    LS = np.zeros(num_symbols)
    LMMSE = np.zeros(num_symbols)
    SCF = np.zeros(num_symbols)
    RVNN = np.zeros(num_symbols)

    # Loop sobre cada símbolo OFDM
    for symbol in range(num_symbols):
        # Demodulação para LS e LMMSE (usam pCE)
        LS_predictedLabels, _ = do_qamdemodulate(rxSymbols_LS[:, symbol], pCE)
        LMMSE_predictedLabels, _ = do_qamdemodulate(rxSymbols_LMMSE[:, symbol], pCE)

        # Demodulação para SCF e RVNN (usam p)
        SCF_predictedLabels, _ = do_qamdemodulate(scfOutput[:, symbol], p)
        RVNN_predictedLabels, _ = do_qamdemodulate(rvNNOuput[:, symbol], p)

        # Cálculo do número de erros (distância de Hamming entre rótulos)
        LS[symbol] = np.sum(LS_predictedLabels != txLabels[:, symbol])
        LMMSE[symbol] = np.sum(LMMSE_predictedLabels != txLabels[:, symbol])
        SCF[symbol] = np.sum(SCF_predictedLabels != XLabels[:, symbol])
        RVNN[symbol] = np.sum(RVNN_predictedLabels != XLabels[:, symbol])

    # Cálculo da taxa de erros (SER) médio
    LS_SER = np.mean(LS)
    LMMSE_SER = np.mean(LMMSE)
    SCF_SER = np.mean(SCF)
    RVNN_SER = np.mean(RVNN)

    return LS_SER, LMMSE_SER, SCF_SER, RVNN_SER
"""
def get_SER_Comparison(p, pCE, txLabels, XLabels, rxSymbols_LS, rxSymbols_LMMSE, scfOutput=None, rvNNOuput=None):
    num_symbols = rxSymbols_LS.shape[1]

    LS = np.zeros(num_symbols)
    LMMSE = np.zeros(num_symbols)
    SCF = np.zeros(num_symbols) if scfOutput is not None else None
    RVNN = np.zeros(num_symbols) if rvNNOuput is not None else None

    for symbol in range(num_symbols):
        # Classical estimation demodulation
        LS_predictedLabels, _ = do_qamdemodulate(rxSymbols_LS[:, symbol], pCE)
        LMMSE_predictedLabels, _ = do_qamdemodulate(rxSymbols_LMMSE[:, symbol], pCE)

        LS[symbol] = np.sum(LS_predictedLabels != txLabels[:, symbol])
        LMMSE[symbol] = np.sum(LMMSE_predictedLabels != txLabels[:, symbol])

        if scfOutput is not None:
            SCF_predictedLabels, _ = do_qamdemodulate(scfOutput[:, symbol], p)
            SCF[symbol] = np.sum(SCF_predictedLabels != XLabels[:, symbol])

        if rvNNOuput is not None:
            RVNN_predictedLabels, _ = do_qamdemodulate(rvNNOuput[:, symbol], p)
            RVNN[symbol] = np.sum(RVNN_predictedLabels != XLabels[:, symbol])

    LS_SER = np.mean(LS)
    LMMSE_SER = np.mean(LMMSE)
    SCF_SER = np.mean(SCF) if SCF is not None else None
    RVNN_SER = np.mean(RVNN) if RVNN is not None else None

    return LS_SER, LMMSE_SER, SCF_SER, RVNN_SER