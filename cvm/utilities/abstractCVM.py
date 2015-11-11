#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


class CVM(object):

    """
    Abstract CVM class
    ====================

    All cvm calculation must inherit this class and
    implement run(self) method
    """

    __slots__ = (
        'count',  # count for self-consistent
        'output',  # output
        'temp',  # temperature
        'chemical_mu',  # opposite chemical potential
        'int_pair',  # pair interaction energy
        'int_trip',  # triple interaction energy
        'int_tetra',  # tetrahedron interaction energy
        'bzc',  # Boltzmann constant
        'x_1',  # elements concentration
        'omega'  # coordination number
    )

    def __init__(self, inp):
        super(CVM, self).__init__()
        self.count = 0
        self.output = {}

        # init
        self.chemical_mu = np.array(inp['chemical_mu'], np.float64)
        self.temp = np.linspace(inp['temp'][0], inp['temp'][1], inp['temp'][2])
        self.bzc = np.float32(inp['bzc'])
        self.int_pair = np.float64(inp['int_en']['pair'])
        self.int_trip = np.float64(inp['int_en']['trip'])
        self.int_tetra = np.float64(inp['int_en']['tetra'])
        self.x_1 = np.float64(inp['x_1'])
        self.omega = np.array(inp['omega'], np.uint8)

    def run(self):
        raise NameError('must implement the inherited abstract method')
        pass
