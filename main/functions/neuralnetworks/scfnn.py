# scfnn.py

import numpy as np

class SCFNN:
    def __init__(self, N, act, eta):
        self.N = N
        self.L = len(N) - 1
        self.act = act
        self.eta = eta
        self.weights = []
        self.biases = []
        self.initialize_parameters()

    def initialize_parameters(self):
        for l in range(self.L):
            w_shape = (self.N[l+1], self.N[l])
            b_shape = (self.N[l+1], 1)
            # Pesos e biases complexos
            self.weights.append((np.random.randn(*w_shape) + 1j * np.random.randn(*w_shape)) * 0.01)
            self.biases.append((np.zeros(b_shape) + 1j * np.zeros(b_shape)))

    def activation(self, x, func):
        if func == 'th':
            return np.tanh(x)
        return x

    def forward(self, x):
        self.a = [x]
        for l in range(self.L):
            z = np.dot(self.weights[l], self.a[-1]) + self.biases[l]
            a = self.activation(z, self.act[l])
            self.a.append(a)
        self.y = self.a[-1]
        return self.y

    def compute_loss(self, y_true):
        loss = np.mean(np.abs(y_true - self.y) ** 2)
        return loss

    def backward(self, y_true):
        self.deltas = [None] * self.L
        # Erro na saída
        self.deltas[-1] = 2 * (self.y - y_true)
        # Erros nas camadas anteriores
        for l in reversed(range(self.L - 1)):
            self.deltas[l] = np.dot(self.weights[l+1].conj().T, self.deltas[l+1]) * (1 - np.abs(self.a[l+1]) ** 2)

    def update_parameters(self):
        for l in range(self.L):
            self.weights[l] -= self.eta[l] * np.dot(self.deltas[l], self.a[l].conj().T)
            self.biases[l] -= self.eta[l] * self.deltas[l]

    def train_step(self, x, y_true):
        self.forward(x)
        loss = self.compute_loss(y_true)
        self.backward(y_true)
        self.update_parameters()
        return loss

    def inference(self, x):
        return self.forward(x)

    def train(self, XTrain, YTrain, XValid, YValid, epochs, valid_freq, show_train_errors, show_validation_errors):
        nBlocksTrain = XTrain.shape[1]
        nBlocksVal = XValid.shape[1]
        mseTrain_SCF = []
        mseVal_SCF = []

        for epoch in range(1, epochs + 1):
            # Embaralhar os dados
            shuffle_indices = np.random.permutation(nBlocksTrain)
            epoch_loss = []

            # Iteração de treinamento
            for idx in shuffle_indices:
                input_SCF = XTrain[:, idx:idx + 1]
                target_SCF = YTrain[:, idx:idx + 1]
                loss = self.train_step(input_SCF, target_SCF)
                epoch_loss.append(loss)

            mseTrain_SCF.append(np.mean(epoch_loss))

            # Exibir erros de treinamento
            if show_train_errors:
                print('=================================================')
                print('Epoch: ', epoch)
                print('Training Errors [dB]:')
                print('     SCF   ')
                print(f'   {10 * np.log10(mseTrain_SCF[-1])}')
                print('=================================================')

            # Validação
            if epoch % valid_freq == 0:
                val_loss = []
                for idx in range(nBlocksVal):
                    input_SCF = XValid[:, idx:idx + 1]
                    target_SCF = YValid[:, idx:idx + 1]
                    output_SCF = self.inference(input_SCF)
                    loss = np.mean(np.abs(target_SCF - output_SCF) ** 2)
                    val_loss.append(loss)
                mseVal_SCF.append(np.mean(val_loss))

                # Exibir erros de validação
                if show_validation_errors:
                    print('=================================================')
                    print('Epoch: ', epoch)
                    print('Validation Errors [dB]')
                    print('[ SCF ] ')
                    print(f'{10 * np.log10(mseVal_SCF[-1])}')
                    print('=================================================')
        return mseTrain_SCF, mseVal_SCF

    def save_model(self, filepath):
        np.save(f'{filepath}_weights.npy', self.weights)
        np.save(f'{filepath}_biases.npy', self.biases)
