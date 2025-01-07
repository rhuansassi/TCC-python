import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest
# Copyright (c) 2016 TU Dresden
# All rights reserved.
# See accompanying license.txt for details.
#
import numpy.testing as nt
import numpy as np

from gfdm.detail.defaultGFDM import get_defaultGFDM
from gfdm.detail import gfdmutil
from gfdm.detail import gabor


class GaborWindowTests(unittest.TestCase):
    def test_small(self):
        p = get_defaultGFDM('small')
        I = gfdmutil.get_receiver_matrix(p, 'ZF')
        gexp = I[0, :]

        g = gfdmutil.get_transmitter_pulse(p)
        gd = gabor.gabdual(g, p.K, p.K)

        nt.assert_array_almost_equal(np.real(gexp), gd)

if __name__ == '__main__':
    unittest.main()
