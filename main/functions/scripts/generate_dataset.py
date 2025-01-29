import numpy as np
import matplotlib.pyplot as plt

from gfdm.detail.Modulator import do_modulate
from gfdm.detail.gfdmutil import do_addcp
from main.functions.gfdm.detail.mllike import *
from scripts.apply_channel_3gpp import apply_channel_3gpp
from scripts.generate_pilots import generate_pilots
from scripts.utils import apply_non_linearities, do_removecp, qammodulate
from main.functions.gfdm.detail.mapping import do_map

def generate_dataset(p, num_symbols, channel, snr_db, h, plot=False):

    train_size = 4 / 5
    s_list = []
    dd_list = []
    yOutCp_list = []


    for k in range(num_symbols):
        # Create data Symbols
        s_k = get_random_symbols_normal(p)
        s_list.append(s_k)

        # QAM Modulation
        M = 2 ** p.mu  # Modulation order
        #dd_k = qammod(s_k, M)

        dd_k = qammodulate(s_k, p)
        dd_list.append(dd_k)

        # Map to D matrix
        Dd = do_map(p, dd_k)

        # Generate Pilots in time Domain
        Dp = generate_pilots(p)

        # Concatenate Data & Pilots
        D = Dd + Dp

        # GFDM - Precode Modulation
        x = do_modulate(p, D)

        # Add cyclic prefix
        x_cp = do_addcp(p, x)

        # Apply Non-Linearities
        y_non_linear = apply_non_linearities(p, x_cp)


        # Receive signal in time
        y = apply_channel_3gpp(channel, y_non_linear, snr_db)  # Corrected unpacking

        # Remove cyclic prefix
        y_out_cp_k = do_removecp(p, y)
        yOutCp_list.append(y_out_cp_k)


    # Save Data
    X = np.column_stack(yOutCp_list)
    Y = np.column_stack(dd_list)
    labels = np.column_stack(s_list)

    num_train = int(num_symbols * train_size)

    x_train = X[:, :num_train]
    y_train = Y[:, :num_train]
    train_labels = labels[:, :num_train]

    x_valid = X[:, num_train:]
    y_valid = Y[:, num_train:]
    valid_labels = labels[:, num_train:]

    return x_train, y_train, x_valid, y_valid, train_labels, valid_labels