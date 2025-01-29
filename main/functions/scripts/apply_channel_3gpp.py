# apply_channel_3gpp.py

import numpy as np


def get_channel_impulse_response(channel):
    max_delay = np.max(channel['delays'])
    h_length = max_delay + 1
    h = np.zeros(h_length, dtype=complex)
    for gain, delay in zip(channel['gains'], channel['delays']):
        h[delay] += gain
    return h

def apply_channel_3gpp(channel, transmitted_signal, SNR_dB):
    # Verificar a dimensão do transmittedSignal
    if transmitted_signal.ndim == 1:
        transmitted_signal = transmitted_signal[:, np.newaxis]
        is_1d_input = True
    else:
        is_1d_input = False

    num_samples, num_blocks = transmitted_signal.shape
    y = np.zeros_like(transmitted_signal, dtype=complex)

    # Obter a resposta ao impulso do canal
    h_channel = get_channel_impulse_response(channel)

    for k in range(num_blocks):
        # Convoluir o sinal transmitido com a resposta ao impulso do canal
        channel_output = np.convolve(transmitted_signal[:, k], h_channel, mode='same')

        # Calcular a potência do sinal
        signal_power = np.mean(np.abs(channel_output) ** 2)

        # Calcular a potência do ruído com base no SNR
        SNR_linear = 10 ** (SNR_dB / 10)
        noise_power = signal_power / SNR_linear

        # Gerar ruído AWGN
        noise = np.sqrt(noise_power / 2) * (
                    np.random.randn(len(channel_output)) + 1j * np.random.randn(len(channel_output)))

        # Adicionar ruído ao sinal de saída do canal
        y[:, k] = channel_output + noise

    # Se o sinal de entrada era 1D, retornar um array 1D
    if is_1d_input:
        y = y[:, 0]

    return y