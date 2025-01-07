import numpy as np
import matplotlib.pyplot as plt

from gfdm.detail.Modulator import do_modulate  # caso seja o modulatePrecode
from gfdm.detail.gfdmutil import do_addcp
from main.functions.gfdm.detail.mllike import *
from scripts.apply_channel_3gpp import apply_channel_3gpp
from scripts.generate_pilots import generate_pilots
from scripts.utils import apply_non_linearities, do_removecp, D_mapPrecode, modulatePrecode
from wlib.qammodulation import qammod

def generate_test_data(p, num_symbols, channel, snr_db, h, plot=False):
    # Pilot positions (0-based)
    pilot_positions = np.arange(0, p.K, p.delta_k)

    # Em vez de pré-alocar arrays, vamos usar listas
    s_list = []
    dd_list = []
    y_list = []
    yNoCp_list = []
    Xp_list = []

    for k in range(num_symbols):
        # Cria símbolos de dados
        s_k = get_random_symbols(p)
        s_list.append(s_k)

        # Modulação QAM
        M = 2 ** p.mu
        dd_k = qammod(s_k, M)
        dd_list.append(dd_k)

        # Mapeia para a matriz D
        Dd = D_mapPrecode(p, dd_k)

        # Gera pilotos no domínio do tempo
        Dp = generate_pilots(p)

        # Concatena dados e pilotos
        D = Dd + Dp

        # Modula a matriz de pilotos (para obter Xp)
        xp = modulatePrecode(p, Dp)
        Xp_k = p.Fp @ xp
        Xp_list.append(Xp_k)

        # Modulação GFDM - Precode
        x = modulatePrecode(p, D)

        # Adiciona CP
        xCp = do_addcp(p, x)

        # Aplica não-linearidades
        yNonLinear = applyNonLinearities(p, xCp)

        # Sinal recebido pelo canal
        y_k = applyChannel3GPP(channel, yNonLinear, snr_db, h)
        y_list.append(y_k)

        # Remove CP
        yOutCp = do_removecp(p, y_k)

        # Remove pilotos
        yOutCp_no_pilots = np.delete(yOutCp, pilot_positions, axis=0)
        yNoCp_list.append(yOutCp_no_pilots)

    s_out = np.array(s_list).T            # (p.K, num_symbols)
    dd_out = np.array(dd_list).T          # (p.K, num_symbols)
    y_out = np.array(y_list).T            # (p.K + p.cpLen, num_symbols)
    yNoCp_out = np.array(yNoCp_list).T    # (p.K - len(pilot_positions), num_symbols)
    Xp_out = np.array(Xp_list).T          # (p.K, num_symbols)

    return s_out, dd_out, y_out, yNoCp_out, Xp_out
