import numpy as np
from main.rosenpy.model.scffnn import SCFFNN
from main.rosenpy.model.rp_optimizer import CVAdamax
from main.rosenpy.utils import act_func


# Função de ativação personalizada para Tangente Hiperbólica (th)
def tanh_activation(module, z, derivative=False):
    if derivative:
        return 1 - module.tanh(z) ** 2
    else:
        return module.tanh(z)


def build_and_train_model(input_dim, output_dim, XTrain_prepared, YTrain_prepared, epochs, batch_size):
    """
    Constrói e treina o modelo SCFFNN com a configuração baseada em parâmetros dados.
    """
    # Cria e configura a rede neural SCFFNN
    nn = SCFFNN(gpu_enable=False)

    # Adiciona camada de entrada com tangente hiperbólica (sem camada oculta)
    #nn.add_layer(neurons=input_dim, ishape=input_dim, activation=tanh_activation)
    nn.add_layer(neurons=input_dim, ishape=input_dim, activation=act_func.tanh)

    # Configura o otimizador com a taxa de aprendizado especificada
    optimizer = CVAdamax()
    optimizer.eta = 0.01  # Define a taxa de aprendizado

    # Treina a rede neural
    nn.fit(
        XTrain_prepared,
        YTrain_prepared,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        verbose=10
    )

    return nn
