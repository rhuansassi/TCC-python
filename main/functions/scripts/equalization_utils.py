import numpy as np

from gfdm.detail.gfdmutil import get_transmitter_pulse, get_receiver_pulse, get_kset, get_mset
from scripts.utils import ifft_u


def LS_Estimation(p, Y, Xp, Fp, delaySpread):

    # Receive signal in time domain
    y = ifft_u(Y)

    # DFT at pilot positions
    Yp = np.dot(Fp, y)

    # LS Estimation
    H = np.divide(Yp, Xp)


    # iFFT at H_LS
    #h_channelLen = (p.delta_k * p.M * np.dot(Fp, H)) / np.sqrt(p.K * p.M)

    h_channelLen = (
            p.delta_k
            * p.M
            * (Fp.conj().T @ H)  # @ é a operação de multiplicação matricial em Python
            / np.sqrt(p.K * p.M)
    )

    # Allocate zeros at positions larger than DelaySpread
    h_channelLen[delaySpread:] = 0

    # Translate to frequency domain
    H = np.fft.fft(h_channelLen)

    # Get wide matrix form
    Hhat = np.diag(H)

    return Hhat


def LMMSE_Estimation(p, SNR_dB, Y, Xp, R_HH, delaySpread, Fp):

    # Noise calculation
    VarS = 1  # Signal power
    VarN = VarS / (10 ** (SNR_dB / 10))  # Noise power
    R_NN = VarN * np.eye(int(p.Kon / p.delta_k))  # Noise covariance matrix

    # Receive signal in time domain
    y = ifft_u(Y)

    # DFT at pilot positions
    Yp = np.dot(Fp, y)

    # Wide matrix pilot form
    dXp = np.diag(Xp)

    # Correlation matrix
    R_YY = np.dot(dXp, np.dot(R_HH, dXp.conj().T)) + R_NN

    H_LMMSE = np.dot(
        np.dot(R_HH, dXp.conj().T),
        np.linalg.solve(R_YY, Yp)
    )

    # FFT interpolation
    #h_lmmse_temp = (p.delta_k * p.M * np.dot(Fp.T, H_LMMSE)) / np.sqrt(p.K * p.M)
    h_lmmse_temp = (
            p.delta_k
            * p.M
            * (Fp.conj().T @ H_LMMSE)
            / np.sqrt(p.K * p.M)
    )

    h_lmmse_temp[delaySpread:] = 0  # Zero padding beyond delay spread

    Hhat = np.fft.fft(h_lmmse_temp)

    # Wide matrix form
    Hhat = np.diag(Hhat)

    return Hhat


def perform_LS_Equalization(p, Y, Hhat, SNR_dB):
    N = p.K * p.M
    SNR_Linear = 1 / (10 ** (SNR_dB / 10))
    Q = np.linalg.solve(
        Hhat.conj().T @ Hhat + SNR_Linear * np.eye(N),
        Hhat.conj().T
    )
    Yeq = Q @ Y.flatten()
    return Yeq



def demodulate_precode(p, x):
    g_tx = get_transmitter_pulse(p)
    G_precode = np.fft.fft(g_tx)
    G_precode[0:2] = 0
    g_precode = np.fft.ifft(G_precode)
    norm_factor = np.sqrt(np.sum(np.abs(g_precode) ** 2))

    g = get_receiver_pulse(p, 'ZF')
    g = g[0::int(p.K / p.K)]
    G = np.fft.fft(g)

    M = p.M
    K = p.K
    L = int(len(G) / M)

    pilot_positions = np.arange(1, K + 1, p.delta_k, dtype=int)
    data_positions = np.setdiff1d(np.arange(1, K + 1, dtype=int), pilot_positions)

    indexes = np.arange(1, M + 1, dtype=int)
    indexes = indexes[1:]
    real_indexes = indexes - 1

    Xhat = np.fft.fft(x)
    Dhat = np.zeros((K, M), dtype=complex)

    for k in pilot_positions:
        shift_amount = int(np.ceil(L * M / 2) - M * (k - 1))
        carrier = np.roll(Xhat, shift_amount)
        portion = carrier[:L * M]
        portion_shifted = np.fft.fftshift(portion)
        carrier_matched = portion_shifted * (G * norm_factor)
        # Reshape em 'Fortran order' para reproduzir o comportamento do reshape do MATLAB
        carrier_matched_2d = carrier_matched.reshape((M, L), order='F')
        dhat = np.sum(carrier_matched_2d, axis=1) / L
        dhat_subset = dhat[real_indexes]
        dhat[real_indexes] = np.fft.ifft(dhat_subset)
        Dhat[k - 1, :] = dhat

    Dhat[pilot_positions - 1, 0] = 0

    for k in data_positions:
        shift_amount = int(np.ceil(L * M / 2) - M * (k - 1))
        carrier = np.roll(Xhat, shift_amount)
        portion = carrier[:L * M]
        portion_shifted = np.fft.fftshift(portion)
        carrier_matched = portion_shifted * G
        carrier_matched = carrier_matched.reshape((M, L))
        dhat = np.sum(carrier_matched, axis=1) / L
        dhat = np.fft.ifft(dhat)
        Dhat[k - 1, :] = dhat

    return Dhat


def unmap_precode(p, Dhat):

    d_set = np.arange(1, p.K * p.M + 1)
    pilot_positions = np.arange(1, p.K + 1, p.delta_k)
    d_set = np.setdiff1d(d_set, pilot_positions)

    kset = get_kset(p) + 1
    mset = get_mset(p) + 1

    if len(kset) == p.K and len(mset) == p.M:
        s_long = Dhat.flatten(order='F')
    else:
        Dm = Dhat[kset - 1][:, mset - 1]
        s_long = Dm.flatten(order='F')

    # dhat = s_long(d_set)  (lembrando que d_set é 1-based)
    dhat = s_long[d_set - 1]

    return dhat


