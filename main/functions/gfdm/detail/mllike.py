import copy

from main.functions.gfdm.detail.gfdmutil import get_kmset, get_noise_enhancement_factor, get_mset, get_kset

import numpy as np

def get_random_symbols_normal(p):

    ms, ks = get_kmset(p)
    num_symbols = len(ms) * len(ks)
    max_value = 2 ** p.mu
    random_symbols = np.random.randint(0, max_value, size=(num_symbols,))

    return random_symbols


def get_random_symbols(p):
    p0 = copy.deepcopy(p)
    p0.M = p.Md + (p.delta_k - 1) * p.M
    p0.Kset = np.arange(1, int(p.Kon / p.delta_k) + 1)

    num = len(get_mset(p0)) * len(get_kset(p0))
    high = 2 ** p.mu

    # Gerar a sequência de símbolos aleatórios
    s = np.random.randint(0, high, size=num)
    return s
