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
