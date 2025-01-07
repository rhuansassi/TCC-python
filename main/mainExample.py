import sys
import os
import time
import numpy as np
import math
import pickle
import tensorflow as tf

from main.functions.gfdm.detail.defaultGFDM import get_defaultGFDM
from main.functions.scripts.utils import get_pilot_frequency, get_eb_n0, get_real_valued_snr, get_num_neurons_per_layer
from main.functions.scripts.generate_channel_model import generate_channel_model
from main.rosenpy.model import SCFFNN
from main.rosenpy.model.rp_optimizer import CVAdamax
from main.rosenpy.utils import act_func
from neuralnetworks.rvnn import create_rvnn, train_rvnn
from neuralnetworks.scfnn import SCFNN
from scripts.generate_dataset_test import generate_dataset_test
from scripts.generate_testdata import generate_test_data
from scripts.generate_dataset import generate_dataset
from scripts.model import build_and_train_model
from scripts.prepare_data import prepare_data_for_scffnn, prepare_labels_for_scffnn, reconstruct_complex_output
from scripts.process_input_nn import process_input_nn

# Início do código
start_time = time.time()

# Controle de exibição de erros
show_train_errors = True
show_validation_errors = False

# Parâmetros de treinamento
do_training = False
num_symbols = 5000
epochs = 20 #70
valid_freq = 10
apply_non_linear = False

# Definição de maxEpochs e iterPerEpoch
maxEpochs = 20 #200
iterPerEpoch = 512

# Parâmetros de comparação
num_symbols_comparison = 1000
num_iterations_per_snr = 11

cp_lengths = [16]
M_orders = [4]
save_dir = './trained-net'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


for mod_order in M_orders:
    for cp_length in cp_lengths:

        # Sinal GFDM
        num_sub_carrier = 64
        num_sub_symbols = 3
        num_pilot_sub_symbols = 1
        pilot_freq = get_pilot_frequency(mod_order)

        # Modulação
        mod_type = 'QAM'
        eb_no_db = get_eb_n0(mod_order)

        # Parâmetros do canal para downlink
        p_dl = get_defaultGFDM('BER')
        p_dl.mod_type = mod_type
        p_dl.K = num_sub_carrier
        p_dl.Mp = 0
        p_dl.Md = num_sub_symbols - p_dl.Mp
        p_dl.M = p_dl.Mp + p_dl.Md
        p_dl.Kon = num_sub_carrier
        p_dl.Ncp = cp_length  # Ajuste aqui
        p_dl.Ncs = 0
        p_dl.matched_window = 0
        p_dl.mu = int(math.log2(mod_order))  # Garantindo valor inteiro
        p_dl.delta_k = 0
        p_dl.apply_non_linearities = apply_non_linear

        # Parâmetros do canal para estimativa de canal (pilotos)
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
        p_ce.mu = int(math.log2(mod_order))  # Garantindo valor inteiro
        p_ce.delta_k = pilot_freq
        p_ce.apply_non_linearities = apply_non_linear

        # Cálculo do SNR
        snr_db = get_real_valued_snr(mod_order)


        channel, h, path_power, delaySpread = generate_channel_model(p_dl, profile='TDLA30')

        # Gerar o conjunto de dados
        XTrain, YTrain, XValid, YValid, TrainLabels, ValidLabels = generate_dataset(p_dl, num_symbols, channel, snr_db, h, plot=False)

        #from scipy.io import loadmat

        # Carrega o arquivo .mat
        #mat_data = loadmat('dataSet.mat')

        # Extrai os arrays individuais
        #XTrain = mat_data['XTrain']
        #YTrain = mat_data['YTrain']
        #XValid = mat_data['XValid']
        #YValid = mat_data['YValid']
        #dataTrain = mat_data['dataTrain']
        #dataValid = mat_data['dataValid']


        # ---------------- RVNN ---------------- #
        NInputs = XTrain.shape[0]
        scalingFactorRVNN = np.sqrt(NInputs)

        XTrainStruct, YTrainStruct, XValidStruct, YValidStruct = process_input_nn(
            XTrain / scalingFactorRVNN,
            YTrain / scalingFactorRVNN,
            XValid / scalingFactorRVNN,
            YValid / scalingFactorRVNN,
            False
        )

        # RVNN Parameters
        inputLayerDim = 2 * NInputs
        numNeuronsPerLayer = get_num_neurons_per_layer(p_dl)

        # Network structure
        rvnn_model = create_rvnn(inputLayerDim, numNeuronsPerLayer)

        if do_training:
            rvnn_history = train_rvnn(
                rvnn_model,
                XTrainStruct,
                YTrainStruct,
                XValidStruct,
                YValidStruct,
                epochs=maxEpochs,
                batch_size=iterPerEpoch,
                valid_freq=valid_freq,
                show_train_errors=show_train_errors,
                show_validation_errors=show_validation_errors
            )

            # Salvar o modelo RVNN treinado
            rvnn_model.save(os.path.join(save_dir, 'trainedNetRVNN.h5'))

        # ---------------- SCFNN ---------------- #
        K = NInputs
        #M = K
        #N = [K, M]  # Sem camada oculta
        act = act_func.tanh  # Função de ativação
        eta = 0.01  # Taxa de aprendizado

        scalingFactorSCF = np.sqrt(NInputs)

        XTrainNorm_SCF = XTrain / scalingFactorSCF
        YTrainNorm_SCF = YTrain / scalingFactorSCF
        XValidNorm_SCF = XValid / scalingFactorSCF
        YValidNorm_SCF = YValid / scalingFactorSCF

        XTrainNorm_SCF = XTrainNorm_SCF.T
        YTrainNorm_SCF = YTrainNorm_SCF.T
        XValidNorm_SCF = XValidNorm_SCF.T
        YValidNorm_SCF = YValidNorm_SCF.T

        netSCF = SCFFNN(gpu_enable=False)
        neurons_SCF = XTrain.shape[0]

        if do_training:
            # Configurar a camada inicial com ishape = 192
            netSCF.add_layer(ishape=NInputs, neurons=NInputs, activation=act_func.tanh, weights_rate=eta, biases_rate=eta)
            netSCF.fit(XTrainNorm_SCF, YTrainNorm_SCF, XValidNorm_SCF, YValidNorm_SCF, epochs=epochs)



        N = p_ce.K * p_ce.M
        F = np.fft.fft(np.eye(N)) / np.sqrt(N)

        freq_pilot_positions = np.arange(0, N, p_ce.delta_k * p_ce.M)
        p_ce.Fp = F[freq_pilot_positions.astype(int), :]

        a = 10 ** (path_power / 10)
        P = a / np.sum(a)
        sqrt_P = np.sqrt(P)
        delaySpread = 12
        zeros_padding = np.zeros(N - delaySpread)
        P_full = np.concatenate([sqrt_P, zeros_padding])

        P_freq = np.sqrt(N) * p_ce.Fp @ np.diag(P_full)
        R_HH = P_freq @ P_freq.conj().T
        Hhat = np.ones((p_ce.K * p_ce.M, num_symbols_comparison))

        LS_SER_It = np.zeros((len(eb_no_db), num_iterations_per_snr))
        LMMSE_SER_It = np.zeros((len(eb_no_db), num_iterations_per_snr))
        SCF_SER_It = np.zeros((len(eb_no_db), num_iterations_per_snr))
        RVNN_SER_It = np.zeros((len(eb_no_db), num_iterations_per_snr))


        for snr_idx, snr_db in enumerate(eb_no_db):
            for iteration in range(num_iterations_per_snr):
                print('=================================================')
                print('           Technique Comparison Process          ')
                print('SNR [dB]:', snr_db)
                print('Iteration:', iteration + 1)
                print('=================================================')

                s, d, y, y_no_cp, Xp = generate_test_data(
                    p_ce, num_symbols_comparison, channel, snr_db, h
                )

                XTest, XLabels = generate_dataset_test(p_dl, num_symbols_comparison, channel, eb_no_db[snr_db], h, plot=False)

a = 1
"""

            for snr_idx, snr_db_value in enumerate(eb_no_db):
                for iteration in range(num_iterations_per_snr):
                    print('=================================================')
                    print('           Technique Comparison Process          ')
                    print('SNR [dB]:', snr_db_value)
                    print('Iteration:', iteration + 1)
                    print('=================================================')

                    # Data generation for testing
                    s, d, y, y_no_cp, Xp = generate_TestData(
                        p_ce, num_symbols_comparison, channel, snr_db_value, h
                    )
                    XTest, XLabels = generate_dataset(
                        p_dl, num_symbols_comparison, channel, snr_db_value, h
                    )

                    # Prepare data for SCFNN
                    XTest_SCF = XTest / scalingFactorSCF
                    XTest_SCF = XTest_SCF.T

                    # Inference with SCFNN
                    scf_output = netSCF.predict(XTest_SCF)
                    scf_output = scf_output.T * scalingFactorSCF

                    # Prepare data for RVNN
                    XTestStruct, _ = process_input_nn(
                        XTest / scalingFactorRVNN, None, None, None, False
                    )
                    XTestStruct = XTestStruct.T

                    # Inference with RVNN
                    rvnn_output = rvnn_model.predict(XTestStruct)

                    # Reconstruct complex output for RVNN
                    rvnn_output_complex = rvnn_output[:, :NInputs] + 1j * rvnn_output[:, NInputs:]
                    rvnn_output_complex = rvnn_output_complex.T * scalingFactorRVNN

                    # Classical Channel Equalization
                    dhat_LS, dhat_LMMSE = channel_equalization(
                        p_ce, y, Xp, Fp, delaySpread, snr_db_value, R_HH
                    )

                    # SER Comparison
                    (
                        LS_SER_It[snr_idx, iteration],
                        LMMSE_SER_It[snr_idx, iteration],
                        SCF_SER_It[snr_idx, iteration],
                        RVNN_SER_It[snr_idx, iteration],
                    ) = get_ser_comparison(
                        p_dl,
                        p_ce,
                        s,
                        XLabels,
                        dhat_LS,
                        dhat_LMMSE,
                        scf_output,
                        rvnn_output_complex,
                    )

            # Calculate average SER results
            N_total = N * num_iterations_per_snr
            SCF_SER = np.sum(SCF_SER_It, axis=1) / (num_iterations_per_snr * N)
            RVNN_SER = np.sum(RVNN_SER_It, axis=1) / (num_iterations_per_snr * N)

            # Discount pilots subcarriers for classical CE - SER
            num_pilots = p_ce.K // p_ce.delta_k
            effective_N = N - num_pilots
            LS_SER = np.sum(LS_SER_It, axis=1) / (num_iterations_per_snr * effective_N)
            LMMSE_SER = np.sum(LMMSE_SER_It, axis=1) / (num_iterations_per_snr * effective_N)

            # Plot and save the results
            plt.figure()
            plt.semilogy(eb_no_db, LMMSE_SER, color='#eeca10', linewidth=2, label='LMMSE')
            plt.semilogy(eb_no_db, LS_SER, color='#18ab2b', linewidth=2, label='LS')
            plt.semilogy(eb_no_db, SCF_SER, color='#429abb', linewidth=2, label='SCFNN')
            plt.semilogy(eb_no_db, RVNN_SER, color='#9142bb', linewidth=2, label='RVNN')
            plt.legend()
            plt.xlabel('SNR [dB]')
            plt.ylabel('SER')
            plt.title(f'SER Comparison - {2 ** p_dl.mu} QAM | {p_dl.Ncp} CP')
            plt.grid(True)
            plt.show()

    # End of the code
    elapsed_time = time.time() - start_time
    print(f"Tempo de execução: {elapsed_time:.2f} segundos")


        # REDES RHUAN

            
       

        # Parâmetros da SCFNN
        K = NInputs
        M = K
        N = [K, M]  # Sem camada oculta
        act = ['th']  # Função de ativação
        eta = [0.01]  # Taxa de aprendizado

        netSCF = SCFNN(N, act, eta)

        scalingFactorSCF = np.sqrt(NInputs)

        XTrainNorm_SCF = XTrain / scalingFactorSCF
        YTrainNorm_SCF = YTrain / scalingFactorSCF
        XValidNorm_SCF = XValid / scalingFactorSCF
        YValidNorm_SCF = YValid / scalingFactorSCF

        if do_training:
            mseTrain_SCF, mseVal_SCF = netSCF.train(
                XTrainNorm_SCF,
                YTrainNorm_SCF,
                XValidNorm_SCF,
                YValidNorm_SCF,
                epochs=500,
                valid_freq=valid_freq,
                show_train_errors=show_train_errors,
                show_validation_errors=show_validation_errors
            )

            # Salvar o modelo SCFNN treinado
            netSCF.save_model('./trained-net/trainedSCF')


    # Finalizando o cronômetro
    elapsed_time = time.time() - start_time
    print(f"Tempo de execução: {elapsed_time:.2f} segundos")

"""