import os
import time
import numpy as np
import math


from matplotlib import pyplot as plt
from main.functions.gfdm.detail.defaultGFDM import get_defaultGFDM
from main.functions.scripts.utils import get_pilot_frequency, get_eb_n0, get_real_valued_snr, get_num_neurons_per_layer
from main.functions.scripts.generate_channel_model import generate_channel_model
from main.rosenpy.model import SCFFNN
from main.rosenpy.utils import act_func
from neuralnetworks.rvnn import create_rvnn, train_rvnn
from scripts.channel_equalization import channel_equalization
from scripts.generate_dataset_test import generate_dataset_test
from scripts.generate_testdata import generate_test_data
from scripts.generate_dataset import generate_dataset
from scripts.get_ser_comparison import get_SER_Comparison
from scripts.perform_inference_networks import perform_inference_networks
from scripts.process_input_nn import process_input_nn


# ----------------------------------------------------------
start_time = time.time()

# Controle de exibição de erros
show_train_errors = True
show_validation_errors = False

# Parâmetros de treinamento
do_training = True
num_symbols = 5000
epochs = 200   #mudar aqui
valid_freq = 10
apply_non_linear = False

# Definição de maxEpochs e iterPerEpoch
max_epochs = 70  #mudar aqui
iter_per_epoch = 512

# Parâmetros de comparação
num_symbols_comparison = 1000
num_iterations_per_snr = 2

cp_lengths = [16]
M_orders = [4]
save_dir = './trained-net'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

simulation_count = 1
for mod_order in M_orders:
    for cp_length in cp_lengths:
        # Sinal GFDM
        num_sub_carrier = 64
        num_sub_symbols = 3
        num_pilot_sub_symbols = 1
        pilot_freq = get_pilot_frequency(mod_order)

        # Modulação
        mod_type = 'QAM'
        eb_no_db = [0,2,4] #get_eb_n0(mod_order)

        # Parâmetros p_dl
        p_dl = get_defaultGFDM('BER')
        p_dl.mod_type = mod_type
        p_dl.K = num_sub_carrier
        p_dl.Mp = 0
        p_dl.Md = num_sub_symbols - p_dl.Mp
        p_dl.M = p_dl.Mp + p_dl.Md
        p_dl.Kon = num_sub_carrier
        p_dl.Ncp = cp_length
        p_dl.Ncs = 0
        p_dl.matched_window = 0
        p_dl.mu = int(math.log2(mod_order))
        p_dl.delta_k = 0
        p_dl.apply_non_linearities = apply_non_linear

        # Parâmetros p_ce
        p_ce = get_defaultGFDM('BER')
        p_ce.mod_type = mod_type
        p_ce.K = num_sub_carrier
        p_ce.Mp = num_pilot_sub_symbols
        p_ce.Md = num_sub_symbols - p_ce.Mp
        p_ce.M = p_ce.Mp + p_ce.Md
        p_ce.Kon = num_sub_carrier
        p_ce.Ncp = cp_length
        p_ce.Ncs = 0
        p_ce.matched_window = 0
        p_ce.mu = int(math.log2(mod_order))
        p_ce.delta_k = pilot_freq
        p_ce.apply_non_linearities = apply_non_linear

        # SNR
        snr_db = get_real_valued_snr(mod_order)

        # Canal
        channel, h, path_power, delay_spread = generate_channel_model(p_dl, profile='TDLA30')

        # Dataset de treinamento/validação
        x_train, y_train, x_valid, y_valid, train_labels, valid_labels = generate_dataset(p_dl, num_symbols, channel, snr_db, h, plot=False)

        # RVNN
        n_inputs = x_train.shape[0]
        scaling_factor_rvnn = np.sqrt(n_inputs)

        x_train_struct, y_train_struct, x_valid_struct, y_valid_struct = process_input_nn(
            x_train / scaling_factor_rvnn,
            y_train / scaling_factor_rvnn,
            x_valid / scaling_factor_rvnn,
            y_valid / scaling_factor_rvnn,
            False
        )

        input_layer_dim = 2 * n_inputs
        num_neurons_per_layer = get_num_neurons_per_layer(p_dl)
        rvnn_model = create_rvnn(input_layer_dim, num_neurons_per_layer)

        if do_training:
            rvnn_history = train_rvnn(
                rvnn_model,
                x_train_struct,
                y_train_struct,
                x_valid_struct,
                y_valid_struct,
                epochs=max_epochs,
                batch_size=iter_per_epoch,
                valid_freq=valid_freq,
                show_train_errors=show_train_errors,
                show_validation_errors=show_validation_errors
            )

        # SCFNN
        scaling_factor_scf = np.sqrt(n_inputs)
        x_train_norm_scf = (x_train / scaling_factor_scf).T
        y_train_norm_scf = (y_train / scaling_factor_scf).T
        x_valid_norm_scf = (x_valid / scaling_factor_scf).T
        y_valid_norm_scf = (y_valid / scaling_factor_scf).T

        netSCF = SCFFNN(gpu_enable=False)
        # Camada simples SCFNN
        netSCF.add_layer(ishape=n_inputs, neurons=n_inputs, activation=act_func.tanh, weights_rate=0.01, biases_rate=0.01)

        if do_training:
            netSCF.fit(x_train_norm_scf, y_train_norm_scf, x_valid_norm_scf, y_valid_norm_scf, epochs=epochs)



        # Preparação LMMSE e LS
        N = p_ce.K * p_ce.M
        F = np.fft.fft(np.eye(N)) / np.sqrt(N)
        freq_pilot_positions = np.arange(1, N + 1, p_ce.delta_k * p_ce.M) - 1
        p_ce.Fp = F[freq_pilot_positions, :]
        a = 10 ** (path_power / 10)
        P = a / np.sum(a)
        delay_spread = 12
        P_full = np.concatenate([np.sqrt(P), np.zeros(N - delay_spread)])
        P_freq = np.sqrt(N) * (p_ce.Fp @ np.diag(P_full))
        R_HH = P_freq @ P_freq.conj().T
        Hhat = np.ones((p_ce.K * p_ce.M, num_symbols_comparison), dtype=complex)

        LS_SER_It = np.zeros((len(eb_no_db), num_iterations_per_snr))
        LMMSE_SER_It = np.zeros((len(eb_no_db), num_iterations_per_snr))
        SCF_SER_It = np.zeros((len(eb_no_db), num_iterations_per_snr))
        RVNN_SER_It = np.zeros((len(eb_no_db), num_iterations_per_snr))


        for snr_idx, snr_value_db in enumerate(eb_no_db):
            for i in range(num_iterations_per_snr):
                print('=================================================')
                print('           Technique Comparison Process          ')
                print('SNR [dB]:', snr_value_db)
                print('Iteration:', i + 1)
                print('=================================================')

                s, d, y, y_no_cp, Xp = generate_test_data(p_ce, num_symbols_comparison, channel, snr_value_db, h)

                # Gera dataset de teste para DL
                XTest, XLabels = generate_dataset_test(p_dl, num_symbols_comparison, channel, snr_value_db, h, plot=False)

                # Inference SCFNN e RVNN
                scfOutput, rvNNOutput = perform_inference_networks(XTest, netSCF, scaling_factor_scf, rvnn_model, scaling_factor_rvnn)

                # Equalização de canal LS e LMMSE
                dhat_LS, dhat_LMMSE = channel_equalization(p_ce, y, Xp, p_ce.Fp, delay_spread, snr_value_db, R_HH)


                # Cálculo do SER
                LS_SER_It[snr_idx, i], LMMSE_SER_It[snr_idx, i], SCF_SER_It[snr_idx, i], RVNN_SER_It[snr_idx, i] = get_SER_Comparison(p_dl, p_ce, s, XLabels, dhat_LS, dhat_LMMSE)#, scfOutput, rvNNOutput)



        SCF_SER = np.sum(SCF_SER_It.T, axis=0) / (num_iterations_per_snr * N)
        RVNN_SER = np.sum(RVNN_SER_It.T, axis=0) / (num_iterations_per_snr * N)


        numPilots = p_ce.K / p_ce.delta_k
        numPilots = int(numPilots)

        LS_SER = np.sum(LS_SER_It.T, axis=0) / (num_iterations_per_snr * (N - numPilots))
        LMMSE_SER = np.sum(LMMSE_SER_It.T, axis=0) / (num_iterations_per_snr * (N - numPilots))


        plt.figure(simulation_count)
        # Plot: note que no MATLAB o primeiro traço é LMMSE, depois LS, etc.
        plt.semilogy(eb_no_db, LMMSE_SER, color='#eeca10', linewidth=2, label='LMMSE')
        plt.semilogy(eb_no_db, LS_SER, color='#18ab2b', linewidth=2, label='LS')
        plt.semilogy(eb_no_db, SCF_SER, color='#429abb', linewidth=2, label='SCFNN')
        plt.semilogy(eb_no_db, RVNN_SER, color='#9142bb', linewidth=2, label='RVNN')

        end_time = time.time()
        print("Tempo total de simulação: ", end_time - start_time, "segundos")

        plt.xlabel('SNR [dB]')
        plt.ylabel('SER')
        plt.title(f'SER Comparison - {2**p_dl.mu} QAM | {p_dl.Ncp} CP')
        plt.grid(True)
        plt.legend()
        plt.show()

