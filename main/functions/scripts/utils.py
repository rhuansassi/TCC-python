import numpy as np
import math
from scipy.signal import fftconvolve
from numpy.random import normal

from gfdm.detail.mapping import do_map
# Adjusted import to match your project structure
from main.functions.gfdm.detail.gfdmutil import get_kset, get_mset, get_transmitter_pulse


def get_pilot_frequency(M):
    if M == 4:
        return 4
    else:
        return 1

def get_eb_n0(mod_order):
    if mod_order == 4:
        return list(range(0, 21, 2))
    elif mod_order == 16:
        return list(range(10, 31, 2))
    else:
        return list(range(15, 36, 2))

def get_real_valued_snr(M):
    if M == 4:
        return 15
    elif M == 16:
        return 18
    else:
        return 20

def apply_non_linearities(p, x):
    if not p.apply_non_linearities:
        return x

    y = np.zeros_like(x, dtype=complex)  # Create y with the same shape as x

    # Define CR value based on 2^p.mu
    M = 2 ** p.mu
    if M == 64:
        CR = 2.25
    else:
        CR = 1.75

    sigma = np.sqrt(np.mean(np.abs(x) ** 2))  # Calculate RMS
    A = CR * sigma  # Threshold limit

    for n in range(len(x)):
        if np.abs(x[n]) <= A:
            y[n] = x[n]
        else:
            y[n] = A * np.exp(1j * np.angle(x[n]))

    return y

def awgn(x, SNR_dB, measured='measured'):
    L = len(x)
    SNR_linear = 10 ** (SNR_dB / 10.0)
    if measured == 'measured':
        signal_power = np.mean(np.abs(x) ** 2)
    else:
        signal_power = 1  # Default signal power
    noise_power = signal_power / SNR_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(L) + 1j * np.random.randn(L))
    return x + noise

"""
def apply_channel_3gpp(channel, transmittedSignal, SNR, h, add_noise=False):
    transmittedSignal = np.array(transmittedSignal)
    if transmittedSignal.ndim == 1:
        transmittedSignal = transmittedSignal.reshape(-1, 1)
    y = np.zeros_like(transmittedSignal, dtype=transmittedSignal.dtype)
    for k in range(transmittedSignal.shape[1]):
        y_channel = channel(transmittedSignal[:, k])
        if add_noise:
            y[:, k] = awgn(y_channel, SNR, measured='measured')
        else:
            y[:, k] = y_channel
    if y.shape[1] == 1:
        y = y.flatten()
    return y, channel, None
"""


def get_window(p):
    # Placeholder function; implement according to your system's model
    # For example, return a Hamming window of the appropriate length
    return np.hamming(p.K * p.M)  # Adjusted to use p.K and p.M

def do_removecp(p, xcp):
    """
    Removes the cyclic prefix from the signal according to parameters in p.
    """
    Ncp = p.NCP  # Corrected attribute name
    Ncs = p.Ncs
    mtchw = p.matched_window

    if mtchw == 0:
        # Without matched window, remove the cyclic prefix directly
        x = xcp[Ncp:(-Ncs if Ncs != 0 else None)]
    else:
        # Ensure p.b is defined
        if not hasattr(p, 'b'):
            # Define p.b based on your system; placeholder value
            p.b = Ncp  # Or set to an appropriate value
        b = p.b

        # Apply the conjugate window
        w = np.conj(get_window(p))
        xcp = xcp * w

        # Adding the roll-off part
        xm = np.copy(xcp)
        xm[0:b] = xcp[0:b] + xcp[-b:]  # Sum the roll-off parts
        xm[-b:] = xm[0:b]  # Restore the correct signal
        x = xm[Ncp:(-Ncs if Ncs != 0 else None)]

    return x


def get_num_neurons_per_layer(p):
    mu_power = 2 ** p.mu

    if mu_power == 4 or mu_power == 16:
        return 1024
    elif mu_power == 64:
        return 2048


def normalize_data(X, scaling_factor):
    return X / scaling_factor

def shuffle_data(X, Y):
    indices = np.random.permutation(X.shape[1])
    return X[:, indices], Y[:, indices]


def D_mapPrecode(p, s):
    subcarrier_positions = np.arange(1, p.M * p.K + 1)  # Índices de 1 a M*K
    pilot_positions = np.arange(1, p.K + 1, p.delta_k)  # Índices espaçados por delta_K

    subcarrier_positions = np.delete(subcarrier_positions, pilot_positions - 1)


    s_long = np.zeros(p.M * p.K, dtype=s.dtype)

    s_long[subcarrier_positions - 1] = s

    # Chama a função do_map
    D = do_map(p, s_long)
    return D


import numpy as np


def modulatePrecode(p, D):
    # Obter o filtro do transmissor
    g = get_transmitter_pulse(p)
    g_dwnsmpl = g[np.round(np.arange(0, p.K, p.K / p.L)).astype(int)]
    G = np.fft.fft(g_dwnsmpl)
    G = np.column_stack((G, G))  # Copiar a coluna para a segunda

    # Filtragem com pré-codificação
    G_precode = np.fft.fft(g)
    G_precode[:2] = [0, 0]
    g_precode = np.fft.ifft(G_precode)
    norm_factor = np.sqrt(np.sum(np.abs(g_precode) ** 2))

    # Renormalizar o filtro para as subportadoras piloto
    G[:, 1] = G[:, 1] / norm_factor
    L = len(G[:, 0]) // p.M

    # === Pré-codificação: FFT apenas em subportadoras de dados (Pilotos Ortogonais) ===
    # Referências iniciais
    pilot_positions = np.arange(1, p.K + 1, p.deltaK)
    data_positions = np.setdiff1d(np.arange(1, p.K + 1), pilot_positions)

    # Transpor e manipular `D`
    Dtemp = D.T
    full_data_indexes = np.arange(1, Dtemp.shape[0])

    # Aplicar FFT em índices de dados
    Dtemp[full_data_indexes, pilot_positions - 1] = np.fft.fft(Dtemp[full_data_indexes, pilot_positions - 1], axis=0)
    Dtemp[:, data_positions - 1] = np.fft.fft(Dtemp[:, data_positions - 1], axis=0)

    # Garantir que o sinal dos pilotos tenha energia igual a 1
    Dtemp[0, pilot_positions - 1] *= (np.sqrt(p.M) / (np.sqrt(p.M - 1) * L * np.max(G[:, 1])))

    # Modulação GFDM
    K, M = p.K, p.M
    N = M * K
    DD = np.tile(Dtemp, (L, 1))
    X = np.zeros(N, dtype=complex)

    for k in range(1, p.K + 1):
        carrier = np.zeros(N, dtype=complex)
        if np.any(np.abs(k - pilot_positions) < 1e-10):
            carrier[:L * M] = np.fft.fftshift(DD[:, k - 1] * G[:, 1])
        else:
            carrier[:L * M] = np.fft.fftshift(DD[:, k - 1] * G[:, 0])
        carrier = np.roll(carrier, -L * M // 2 + M * (k - 1))
        X += carrier

    X = (K / L) * X
    x = np.fft.ifft(X)
    return x

