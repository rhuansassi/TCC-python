import numpy as np
import math
from scipy.signal import fftconvolve
from numpy.random import normal

from gfdm.detail.mapping import do_map
# Adjusted import to match your project structure
from main.functions.gfdm.detail.gfdmutil import get_kset, get_mset, get_transmitter_pulse
from wlib.qammodulation import qamdemod


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
    return np.hamming(p.K * p.M)  # Adjusted to use p.K and p.M

def do_removecp(p, xcp):
    Ncp = p.Ncp
    Ncs = p.Ncs
    b = p.b
    mtchw = p.matched_window


    if mtchw == 0:
        if Ncs > 0:
            x = xcp[Ncp:-Ncs]
        else:
            x = xcp[Ncp:]
    else:
        w = np.conjugate(get_window(p))
        xcp = xcp * w

        xm = np.copy(xcp)
        xm[:b] = xcp[:b] + xcp[-b:]
        xm[-b:] = xm[:b]

        # Remove CP e CS
        if Ncs > 0:
            x = xm[Ncp:-Ncs]
        else:
            x = xm[Ncp:]

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


def modulatePrecode(p, D):
    # Obtém o pulso do transmissor
    g = get_transmitter_pulse(p)

    # Índices de amostragem para downsampling
    step = p.K / p.L
    indices = np.arange(1, len(g) + 1, step)  # MATLAB: 1:p.K/p.L:end
    indices = np.round(indices).astype(int) - 1  # Ajuste para zero-based
    g_dwnsmpl = g[indices]

    # FFT do pulso amostrado
    G = np.fft.fft(g_dwnsmpl)
    # Cria segunda coluna igual à primeira
    G = np.column_stack((G, G))

    # Pré-codificação do filtro G
    G_precode = np.fft.fft(g)
    G_precode[0:2] = 0  # Zera os dois primeiros elementos
    g_precode = np.fft.ifft(G_precode)
    norm_factor = np.sqrt(np.sum(np.abs(g_precode) ** 2))

    # Normaliza filtro de piloto
    G[:, 1] = G[:, 1] / norm_factor

    # Calcula L
    L = int(G.shape[0] / p.M)

    # Posições de piloto (0-based)
    pilotPositions = np.arange(0, p.K, p.delta_k)

    # Posições de dados
    dataPositions = np.arange(p.K)
    dataPositions = dataPositions[~np.in1d(dataPositions, pilotPositions)]

    # Transpõe D (Dtemp = D.')
    Dtemp = D.T

    # fullDataIndexes: Em MATLAB era 2:M, assumindo Dtemp é (M x K)
    # Se Dtemp é (M x K), então M = Dtemp.shape[0]
    M = p.M
    fullDataIndexes = np.arange(1, M)  # [1, 2, ..., M-1]

    # Aplica FFT nas posições de piloto nos índices fullDataIndexes (ao longo das linhas => axis=0)
    Dtemp[fullDataIndexes[:, None], pilotPositions] = np.fft.fft(Dtemp[fullDataIndexes[:, None], pilotPositions],
                                                                 axis=0)

    # Aplica FFT nas posições de dados (ao longo das linhas => axis=0)
    Dtemp[:, dataPositions] = np.fft.fft(Dtemp[:, dataPositions], axis=0)

    # Ajuste de energia nos pilotos
    # Nota: Em MATLAB, Dtemp(1, pilotPositions) é a primeira linha => Dtemp[0, pilotPositions] no Python
    Dtemp[0, pilotPositions] = Dtemp[0, pilotPositions] * (np.sqrt(p.M) / (np.sqrt(p.M - 1) * L * np.max(G[:, 1])))

    # GFDM Modulação
    K = p.K
    N = M * K
    # Repete Dtemp L vezes nas linhas
    DD = np.tile(Dtemp, (L, 1))
    X = np.zeros(N, dtype=complex)

    # Laço para compor o sinal
    for k in range(K):
        carrier = np.zeros(N, dtype=complex)
        # Verifica se k é posição de piloto
        if k in pilotPositions:
            # Pilotos usam coluna 1 de G (zero-based = G[:,1])
            carrier[0:L * M] = np.fft.fftshift(DD[:, k] * G[:, 1])
        else:
            # Dados usam coluna 0 de G (zero-based = G[:,0])
            carrier[0:L * M] = np.fft.fftshift(DD[:, k] * G[:, 0])

        # circshift equivalente a np.roll
        shift_amount = - (L * M) // 2 + M * k
        carrier = np.roll(carrier, shift_amount)
        X += carrier

    # Normalização final
    X = (K / L) * X

    # IFFT final
    x = np.fft.ifft(X)
    return x


def fft_u(x):
    return np.fft.fft(x) / np.sqrt(x.shape[0])


def ifft_u(x):

    return np.fft.ifft(x) * np.sqrt(len(x))


import numpy as np

def do_qamdemodulate(d, p):

    # Reescale de acordo com o fator sqrt(2/3*(2^mu - 1))
    d = d * np.sqrt((2/3) * (2** p.mu - 1))

    if p.mod_type == 'QAM':
        # Se você tiver uma função qamdemod customizada ou de biblioteca, use-a.
        # Aqui, chamamos uma função fictícia (que você deverá implementar ou
        # substituir pela chamada correta de alguma biblioteca).
        sh = qamdemod(d, 2**p.mu)

    elif p.mod_type == 'PSK':
        # Idem para a demodulação PSK
        sh = pskdemod(d, 2**p.mu)

    else:
        raise ValueError(f"Tipo de modulação {p['modType']} não suportado.")

    return sh, d


