#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
CVM data class
==============

data class for data storage and share
"""
import numpy as np


class data(object):

    """data storage class"""

    __slots__ = ('inp',  # INCAR
                 'output',  # output data
                 'temp',  # temperature
                 'mu_ij',  # opposite chemical potential
                 'e_ij',  # interaction energy
                 'k',  # Boltzmann constant
                 'x_i',  # elements concentration
                 'omega'  # coordination number
                 )

    def __init__(self, inp):
        super(data, self).__init__()
        self.inp = inp  # copy inpcard
        self.output = {}

        # init
        self.mu_ij = np.float32(-0.0)
        self.temp = np.intc(0)
        self.k = np.float32(inp['bzc'])
        self.e_ij = np.matrix(inp['int_en']['pair'], np.float32)
        self.x_i = np.matrix(inp['c'], np.float32)
        self.omega = np.matrix(inp['omega'], dtype=np.uint8)
