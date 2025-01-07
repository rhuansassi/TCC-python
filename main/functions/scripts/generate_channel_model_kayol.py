# generate_channel_model.py

import numpy as np
from scripts.channel_tdl import channel_tdl

def generate_channel_model(p, profile='TDL-A'):
    """
    Gera o modelo de canal usando os ganhos e atrasos do channel_tdl.

    Parâmetros:
        p (object): Objeto contendo os parâmetros do sistema (p.K e p.M).
        profile (str): Perfil do canal ('TDL-A', 'TDL-B', etc.).

    Retorna:
        channel (dict): Dicionário contendo os parâmetros do canal.
        h (int): Placeholder (pode ser usado conforme necessário).
        pathPower_dB (numpy.ndarray): Ganhos dos caminhos em dB.
        rms_delay_spread (float): Espalhamento de atraso RMS calculado.
    """

    # Definir o espalhamento de atraso RMS (em segundos)
    if profile == 'TDL-A':
        delay_spread = 30e-9  # 30 ns
    elif profile == 'TDL-B':
        delay_spread = 100e-9  # 100 ns
    elif profile == 'TDL-C':
        delay_spread = 300e-9  # 300 ns
    else:
        raise ValueError('Perfil de canal desconhecido.')

    # Calcular a taxa de amostragem
    subCarrierSpacing = 240e3
    sampleRate = (p.K * p.M) * subCarrierSpacing
    ts = 1 / sampleRate

    # Obter o canal usando o channel_tdl
    channel = channel_tdl(ts, delay_spread, profile)

    # Calcular o espalhamento de atraso RMS
    delays_in_seconds = channel['delays'] * ts
    gains_linear = channel['gains']
    power_linear = np.abs(gains_linear) ** 2

    mean_delay = np.sum(power_linear * delays_in_seconds) / np.sum(power_linear)
    rms_delay_spread = np.sqrt(np.sum(power_linear * (delays_in_seconds - mean_delay) ** 2) / np.sum(power_linear))

    # Obter os ganhos em dB
    pathPower_dB = 20 * np.log10(np.abs(gains_linear))

    h = 1  # Placeholder

    return channel, h, pathPower_dB, rms_delay_spread
