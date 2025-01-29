import numpy as np
import math
from scipy.signal import fftconvolve
from numpy.random import normal

from gfdm.detail.mapping import do_map
# Adjusted import to match your project structure
from main.functions.gfdm.detail.gfdmutil import get_kset, get_mset, get_transmitter_pulse, get_receiver_pulse
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


def D_map_precode(p, s):
    subcarrier_positions = np.arange(1, p.M * p.K + 1)  # Índices de 1 a M*K
    pilot_positions = np.arange(1, p.K + 1, p.delta_k)  # Índices espaçados por delta_K

    subcarrier_positions = np.delete(subcarrier_positions, pilot_positions - 1)


    s_long = np.zeros(p.M * p.K, dtype=s.dtype)

    s_long[subcarrier_positions - 1] = s

    # Chama a função do_map
    D = do_map(p, s_long)
    return D


def modulate_precode(p, D):
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


def do_qamdemodulate(y, p):

    M = 2 ** p.mu
    sqrtM = int(np.sqrt(M))
    if sqrtM * sqrtM != M:
        raise ValueError("Demodulador requer QAM retangular: M deve ser um quadrado perfeito.")

    # Mesma escala usada na modulação
    scale = np.sqrt((2/3) * (M - 1))

    # Moldamos y para 1D para processar facilmente
    original_shape = y.shape
    y_flat = y.ravel()

    # Desnormaliza
    y_scaled = y_flat * scale
    y_real = y_scaled.real
    y_imag = y_scaled.imag

    def quantize_dim(val):
        # i_approx = round((val + (sqrtM-1))/2)
        i_approx = int(np.round((val + (sqrtM - 1)) / 2.0))
        if i_approx < 0:
            i_approx = 0
        elif i_approx > sqrtM - 1:
            i_approx = sqrtM - 1
        return i_approx

    def gray2bin(g):
        b = 0
        while g > 0:
            b ^= g
            g >>= 1
        return b

    # Vetorizar as funções acima para operar em arrays
    qdim_vec = np.vectorize(quantize_dim)
    gray2bin_vec = np.vectorize(gray2bin)

    # i_gray e q_gray => arrays de mesmo tamanho que y_flat
    i_gray = qdim_vec(y_real)
    q_gray = qdim_vec(y_imag)

    # Converte cada i_gray, q_gray em i_bin, q_bin
    i_bin = gray2bin_vec(i_gray)
    q_bin = gray2bin_vec(q_gray)

    # Símbolo final: s = q_bin* sqrtM + i_bin
    s_hat_flat = q_bin * sqrtM + i_bin

    # Retorna ao formato original
    s_hat = s_hat_flat.reshape(original_shape)

    return s_hat

def demodulate_precode(p, x):
    # Get Precode Matrix
    g_tx = get_transmitter_pulse(p)
    G_precode = np.fft.fft(g_tx)
    G_precode[:2] = [0, 0]
    g_precode = np.fft.ifft(G_precode)
    norm_factor = np.sqrt(np.sum(np.abs(g_precode) ** 2))

    # Get Zero Forcing Receiver Pulse
    g = get_receiver_pulse(p, 'ZF')
    g = g[::p.K // p.K]  # Equivalent to MATLAB's g(1:p.K/p.K:end)
    G = np.fft.fft(g)

    # Parameters
    M, K = p.M, p.K
    L = len(G) // M

    # Initialize references
    if p.delta_k > 0:
        pilot_positions = np.arange(0, p.K, p.delta_k)
    else:
        pilot_positions = np.array([])  # Lista vazia

    data_positions = np.setdiff1d(np.arange(0, p.K), pilot_positions)

    indexes = np.arange(1, M)  # Equivalent to MATLAB's indexes = 1:M; indexes(1) = [];

    Xhat = np.fft.fft(x)
    Dhat = np.zeros((K, M), dtype=complex)

    # At pilot subcarriers apply Unnormalized Filter
    if pilot_positions.size > 0:
        for k in pilot_positions:
            carrier = np.roll(Xhat, np.ceil(L * M / 2).astype(int) - M * (k - 1))
            carrier = np.fft.fftshift(carrier[:L * M])
            carrier_matched = carrier * (G * norm_factor)
            dhat = np.sum(carrier_matched.reshape(M, L), axis=1) / L
            dhat[indexes] = np.fft.ifft(dhat[indexes])
            Dhat[k, :] = dhat

    # Set zeros on pilot positions only if pilot_positions is not empty
    if pilot_positions.size > 0:
        Dhat[pilot_positions, 0] = np.zeros(len(pilot_positions))

    # At data positions demodulate normally
    for k in data_positions:
        carrier = np.roll(Xhat, np.ceil(L * M / 2).astype(int) - M * (k - 1))
        carrier = np.fft.fftshift(carrier[:L * M])
        carrier_matched = carrier * G
        dhat = np.sum(carrier_matched.reshape(M, L), axis=1) / L
        dhat = np.fft.ifft(dhat)
        Dhat[k, :] = dhat

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


def qammodulate(s, p):

    s = np.asarray(s, dtype=int)

    M = 2 ** p.mu

    if np.any(s < 0) or np.any(s >= M):
        raise ValueError(f"Símbolos em 's' devem estar em [0..{M-1}].")

    sqrtM = int(np.sqrt(M))
    if sqrtM * sqrtM != M:
        raise ValueError("Para QAM retangular, 2^p.mu deve ser um quadrado perfeito (ex.: 16, 64).")

    def bin2gray(n):
        return n ^ (n >> 1)

    table = np.zeros(M, dtype=int)
    idx = 0
    for q_bin in range(sqrtM):
        for i_bin in range(sqrtM):
            i_gray = bin2gray(i_bin)
            q_gray = bin2gray(q_bin)
            final_index = q_gray * sqrtM + i_gray
            table[idx] = final_index
            idx += 1

    s_gray = table[s]

    I = s_gray % sqrtM
    Q = s_gray // sqrtM


    I2 = 2 * I - sqrtM + 1
    Q2 = 2 * Q - sqrtM + 1

    # Símbolo complexo
    d = I2 + 1j * Q2


    scale = np.sqrt((2/3) * (M - 1))
    d = d / scale

    return d


def get_kset(p):
    if hasattr(p, 'Kset'):
        kset = np.mod(p.Kset, p.K)
        assert len(kset) <= p.K
    elif hasattr(p, 'Kon'):

        up_to = int(np.ceil(p.Kon / 2))
        first_part = np.arange(1, up_to + 1)
        down_from = p.K + 1 - int(np.floor(p.Kon / 2))
        second_part = np.arange(down_from, p.K + 1)
        kset = np.concatenate([first_part, second_part]) - 1
        assert p.Kon <= p.K
    else:
        # se não houver Kset ou Kon, assuma todos subcarriers
        kset = np.arange(0, p.K)

    assert len(np.unique(kset)) == len(kset), "kset contém duplicados!"
    return kset


def get_mset(p):
    if hasattr(p, 'Mset'):
        mset = np.mod(p.Mset, p.M)
        assert len(mset) <= p.M
    elif hasattr(p, 'Mon'):
        # mset = ceil((p.M-p.Mon)/2) : (p.M-1-floor((p.M-p.Mon)/2))
        start = int(np.ceil((p.M - p.Mon) / 2))
        end = p.M - 1 - int(np.floor((p.M - p.Mon) / 2))
        mset = np.arange(start, end + 1)
        assert p.Mon <= p.M
    else:
        # se não houver Mset ou Mon, assuma todos subsymbols
        mset = np.arange(0, p.M)

    assert len(np.unique(mset)) == len(mset), "mset contém duplicados!"
    return mset


def do_map(p, s):

    kset = get_kset(p)
    mset = get_mset(p)

    # Caso todos subcarriers e subsymbols estejam ativos
    if len(kset) == p.K and len(mset) == p.M:
        # Em MATLAB: D = reshape(s, [p.K, p.M]) (column-major)
        D = s.reshape((p.K, p.M), order='F')
    else:
        # Caso apenas parte deles esteja ativa
        # Dm = reshape(s, [length(kset), length(mset)])
        Dm = s.reshape((len(kset), len(mset)), order='F')

        # res1(kset, :) = Dm
        res1 = np.zeros((p.K, len(mset)), dtype=complex)
        res1[kset, :] = Dm

        # res(:, mset) = res1
        res = np.zeros((p.K, p.M), dtype=complex)
        res[:, mset] = res1

        D = res

    return D


