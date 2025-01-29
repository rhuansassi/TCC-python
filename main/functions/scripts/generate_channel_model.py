import numpy as np
from scripts.channel_tdl_custom import channel_tdl_custom

def generate_channel_model(p, profile='TDLA30'):
    if profile == 'TDLA30':
        # TDL-A com DS = 30 ns
        path_power_d_b = np.array([
            -15.5,  0,  -5.1, -5.1, -9.6,
            -8.2,  -13.1, -11.5, -11, -16.2,
            -16.6, -26.2
        ])

        # Em segundos (12 caminhos, iguais ao seu MATLAB)
        path_delays_in_seconds = np.array([
            0.0e-9,   1.0e-8,   1.5e-8,   2.0e-8,
            2.5e-8,   5.0e-8,   6.5e-8,   7.5e-8,
            1.05e-7,  1.35e-7,  1.50e-7,  2.90e-7
        ])

        # RMS Delay Spread nominal
        delaySpread = 30e-9  # 30 ns

    elif profile == 'TDLB100':
        # TDL-B com DS = 100 ns (exemplo)
        path_power_d_b = np.array([
            0, -2.2, -0.6, -0.6, -0.3, -1.2,
            -5.9, -2.2, -0.8, -6.3, -7.5, -7.1
        ])
        path_delays_in_seconds = 1e-9 * np.array([
            0, 10, 20, 30, 35, 45, 55, 120, 170, 245, 330, 480
        ])
        delaySpread = 100e-9

    elif profile == 'TDLC300':
        # TDL-C com DS = 300 ns (exemplo)
        path_power_d_b = np.array([
            -6.9, 0, -7.7, -2.5, -2.4,
            -9.9, -8.0, -6.6, -7.1, -13.0,
            -14.2, -16.0
        ])
        path_delays_in_seconds = 1e-9 * np.array([
            0, 65, 70, 190, 195, 200,
            240, 325, 520, 1045, 1510, 2595
        ])
        delaySpread = 300e-9

    else:
        raise ValueError('Perfil de canal desconhecido.')


    sample_rate = 4.608e6
    ts = 1 / sample_rate

    # Cria o modelo de canal chamando a função customizada
    channel = channel_tdl_custom(
        ts,
        path_delays_in_seconds,
        path_power_d_b
    )

    h = 1

    return channel, h, path_power_d_b, delaySpread

