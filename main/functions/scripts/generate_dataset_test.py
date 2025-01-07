import numpy as np
import matplotlib.pyplot as plt

from gfdm.detail.Modulator import do_modulate
from gfdm.detail.gfdmutil import do_addcp
from main.functions.gfdm.detail.mllike import *
from scripts.apply_channel_3gpp import apply_channel_3gpp
from scripts.generate_pilots import generate_pilots
from scripts.utils import apply_non_linearities, do_removecp
from wlib.qammodulation import qammod
from main.functions.gfdm.detail.mapping import do_map

def generate_dataset_test(p, num_symbols, channel, snr_db, h, plot=False):

    train_size = 5 / 5
    s_list = []
    dd_list = []
    yOutCp_list = []


    for k in range(num_symbols):
        # Create data Symbols
        s_k = get_random_symbols_normal(p)
        s_list.append(s_k)

        # QAM Modulation
        M = 2 ** p.mu  # Modulation order
        dd_k = qammod(s_k, M)
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
        xCp = do_addcp(p, x)

        # Apply Non-Linearities
        yNonLinear = apply_non_linearities(p, xCp)


        # Receive signal in time
        y = apply_channel_3gpp(channel, yNonLinear, snr_db)  # Corrected unpacking

        # Remove cyclic prefix
        yOutCp_k = do_removecp(p, y)
        yOutCp_list.append(yOutCp_k)


    # Save Data
    X = np.column_stack(yOutCp_list)
    Y = np.column_stack(dd_list)
    labels = np.column_stack(s_list)

    num_train = int(num_symbols * train_size)

    Xtest = X[:, :num_train]
    TestLabels = labels[:, :num_train]



    return Xtest, TestLabels
