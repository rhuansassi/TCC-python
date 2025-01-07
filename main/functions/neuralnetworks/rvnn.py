

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np


class PrintDBLoss(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            loss = logs.get('loss')
            val_loss = logs.get('val_loss')
            if loss is not None:
                loss_db = 10 * np.log10(loss)
                print(f'Epoch {epoch + 1}: Training Loss [dB]: {loss_db:.4f}')
            if val_loss is not None:
                val_loss_db = 10 * np.log10(val_loss)
                print(f'Epoch {epoch + 1}: Validation Loss [dB]: {val_loss_db:.4f}')


def create_rvnn(input_dim, num_neurons):
    # Construir o modelo
    model = models.Sequential(name='RVNN')
    model.add(layers.InputLayer(input_shape=(input_dim,), name='input'))
    model.add(layers.Dense(num_neurons, activation='tanh', name='linear1'))
    model.add(layers.Dense(input_dim, name='linearOutput'))

    # Compilar o modelo
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='mean_squared_error')
    return model


def train_rvnn(model, XTrain, YTrain, XValid, YValid, epochs, batch_size, valid_freq, show_train_errors,
               show_validation_errors):
    # Definir os callbacks
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.98,
        patience=5,
        verbose=1
    )
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    print_db_loss = PrintDBLoss()  # Nosso callback personalizado

    # Lista de callbacks
    callback_list = [reduce_lr, early_stop]

    # Adicionar o callback de impressão de erro em dB se show_train_errors estiver ativado
    if show_train_errors or show_validation_errors:
        callback_list.append(print_db_loss)

    # Treinar o modelo
    history = model.fit(
        XTrain.T,
        YTrain.T,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(XValid.T, YValid.T),
        callbacks=callback_list,
        verbose=1
    )
    return model, history
