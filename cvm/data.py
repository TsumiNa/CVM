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
                 'mu',  # opposite chemical potential
                 'eij',  # interaction energy
                 'k',  # Boltzmann constant
                 )

    def __init__(self, inp):
        super(data, self).__init__()
        self.inp = inp
        self.output = {}
        self.temp = np.intc(0)
        self.k = inp[]
        self.mu = np.float32(0.0)
        self.eij = np.float32(inp['pair'])
