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
                 'int_pair',  # pair interaction energy
                 'int_trip',  # triple interaction energy
                 'int_tetra',  # tetrahedron interaction energy
                 'bzc',  # Boltzmann constant
                 'x_1',  # elements concentration
                 'omega'  # coordination number
                 )

    def __init__(self, inp):
        super(data, self).__init__()
        self.inp = inp  # copy inpcard
        self.output = {}
        self.mu_ij = np.float64(0.00)

        # init
        self.temp = np.linspace(inp['temp'][0], inp['temp'][1], inp['temp'][2])
        self.bzc = np.float32(inp['bzc'])
        self.int_pair = np.float64(inp['int_en']['pair'][0])
        self.int_trip = np.float64(inp['int_en']['trip'][0])
        self.int_tetra = np.float64(inp['int_en']['tetra'][0])
        self.x_1 = np.float64(inp['x_1'])
        self.omega = np.array(inp['omega'], dtype=np.uint8)
