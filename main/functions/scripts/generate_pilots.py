import numpy as np

from gfdm.detail.gfdmutil import get_kset


class PilotParameters:
    def __init__(self, delta_k, Kon, K, M):
        self.deltaK = delta_k  # Intervalo de subportadora para os pilotos
        self.Kon = Kon        # Número de subportadoras ativas
        self.K = K            # Número total de subportadoras
        self.M = M            # Número de símbolos no domínio do tempo

def zadoff_chu_seq(root, num_symbols):

    n = np.arange(num_symbols)
    zc_sequence = np.exp(-1j * np.pi * root * n * (n + 1) / num_symbols)
    return zc_sequence

def generate_pilots(p):
    if p.delta_k == 0:
        return 0

    subcarrier_positions = get_kset(p) + 1
    num_pilots = int(np.floor(p.Kon / p.delta_k)) + 1

    pilot_symbols = zadoff_chu_seq(1, num_pilots)


    Dp = np.zeros((p.K, p.M), dtype=complex)

    Dp[subcarrier_positions[::p.delta_k], 0] = pilot_symbols[:int(np.ceil(p.Kon / p.delta_k))]

    return Dp
