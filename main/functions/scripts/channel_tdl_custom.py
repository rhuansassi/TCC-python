# channel_tdl_custom.py

import numpy as np

def channel_tdl_custom(ts, pathDelaysInSeconds, pathPower_dB):
    # Converter ganhos de dB para linear (amplitude)
    lin_gains = 10 ** (pathPower_dB / 20)

    # Normalizar ganhos para que a soma das potências seja 1
    lin_gains /= np.sqrt(np.sum(np.abs(lin_gains) ** 2))

    # Converter atrasos em segundos para atrasos em número de amostras
    s_delays = np.round(pathDelaysInSeconds / ts).astype(int)

    # Obter atrasos únicos e somar ganhos correspondentes
    unique_delays, indices = np.unique(s_delays, return_inverse=True)
    gains_combined = np.zeros(len(unique_delays), dtype=complex)

    for i, idx in enumerate(indices):
        gains_combined[idx] += lin_gains[i]

    # Montar o dicionário do canal
    channel_dic = {
        'gains': gains_combined,
        'delays': unique_delays,
        'los': False,
        'k_factor': 0,           # Fator K para canal Rayleigh
        'los_power': 0,
        'mean_los_power': 0
    }

    return channel_dic
