import numpy as np

from scripts.equalization_utils import LS_Estimation, LMMSE_Estimation, perform_LS_Equalization, demodulate_precode, \
    unmap_precode
from scripts.utils import do_removecp, fft_u, ifft_u


def channel_equalization(p, y, Xp, Fp, delaySpread, SNR_dB, R_HH):

    n_cols = y.shape[1]

    # Listas temporárias para armazenar as colunas (pacotes) equalizadas
    dhat_LS_list = []
    dhat_LMMSE_list = []

    # Loop para cada pacote/coluna
    for k in range(n_cols):
        # 1) Remover o CP
        yNoCp = do_removecp(p, y[:, k])

        # 2) Converter para domínio da frequência
        Y = fft_u(yNoCp)

        # 3) Estimação do canal (LS e LMMSE)
        Hhat_LS = LS_Estimation(p, Y, Xp, Fp, delaySpread)
        Hhat_LMMSE = LMMSE_Estimation(p, SNR_dB, Y, Xp, R_HH, delaySpread, Fp)

        # 4) Equalização (MMSE)
        Xhat_LS = perform_LS_Equalization(p, Y, Hhat_LS, SNR_dB)
        Xhat_LMMSE = perform_LS_Equalization(p, Y, Hhat_LMMSE, SNR_dB)

        # 5) Converter de volta para o domínio do tempo
        xhat_LS = ifft_u(Xhat_LS)
        xhat_LMMSE = ifft_u(Xhat_LMMSE)

        # 6) Demodular (ex.: remoção de GFDM ou pré-codificação)
        DHat_LS = demodulate_precode(p, xhat_LS)
        DHat_LMMSE = demodulate_precode(p, xhat_LMMSE)

        # 7) "Desmapear" a matriz D em símbolos
        dhat_LS_k = unmap_precode(p, DHat_LS)
        dhat_LMMSE_k = unmap_precode(p, DHat_LMMSE)

        # Armazena o resultado do pacote k
        dhat_LS_list.append(dhat_LS_k)
        dhat_LMMSE_list.append(dhat_LMMSE_k)

    # Empilhar as colunas resultantes
    dhat_LS = np.column_stack(dhat_LS_list)
    dhat_LMMSE = np.column_stack(dhat_LMMSE_list)

    return dhat_LS, dhat_LMMSE