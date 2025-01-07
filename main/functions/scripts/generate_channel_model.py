# generate_channel_model.py

import numpy as np

from scripts.channel_tdl_custom import channel_tdl_custom


def generate_channel_model(p, profile='TDLA30'):
    if profile == 'TDLA30':
        # TDLA30 (DS = 30ns)
        pathPower_dB = np.array([-15.5, 0, -5.1, -5.1, -9.6, -8.2, -13.1,
                                 -11.5, -11, -16.2, -16.6, -26.2])
        pathDelaysInSeconds = 1e-9 * np.array([0, 10, 15, 20, 25, 50, 65, 75,
                                               105, 135, 150, 290])
        delaySpread = 12e-9  # 12 ns
    elif profile == 'TDLB100':
        # TDLB100 (DS = 100ns)
        pathPower_dB = np.array([0, -2.2, -0.6, -0.6, -0.3, -1.2, -5.9,
                                 -2.2, -0.8, -6.3, -7.5, -7.1])
        pathDelaysInSeconds = 1e-9 * np.array([0, 10, 20, 30, 35, 45, 55,
                                               120, 170, 245, 330, 480])
        delaySpread = 100e-9  # 100 ns
    elif profile == 'TDLC300':
        # TDLC300 (DS = 300ns)
        pathPower_dB = np.array([-6.9, 0, -7.7, -2.5, -2.4, -9.9, -8.0,
                                 -6.6, -7.1, -13.0, -14.2, -16.0])
        pathDelaysInSeconds = 1e-9 * np.array([0, 65, 70, 190, 195, 200,
                                               240, 325, 520, 1045, 1510, 2595])
        delaySpread = 300e-9  # 300 ns
    else:
        raise ValueError('Perfil de canal desconhecido.')

    # Calcular a taxa de amostragem
    subCarrierSpacing = 240e3
    sampleRate = (p.K * p.M) * subCarrierSpacing
    ts = 1 / sampleRate

    # Gerar o modelo de canal usando 'channel_tdl_custom' com parâmetros personalizados
    channel = channel_tdl_custom(ts, pathDelaysInSeconds, pathPower_dB)
    h = 1

    return channel, h, pathPower_dB, delaySpread
