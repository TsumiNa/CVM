#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import datetime as dt


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
        'bzc',  # Boltzmann constant
        'x_1',  # elements concentration
        'delta_mu',  # opposite chemical potential
        'int_pair',  # pair interaction energy
        'int_trip',  # triple interaction energy
        'int_tetra'  # tetrahedron interaction energy
    )

    def __init__(self, inp):
        super(CVM, self).__init__()
        self.count = 0
        self.output = {}
        self.output['Meta'] = {}
        self.output['Results'] = []

        ##################
        # init
        ##################

        # chemical potential
        if len(inp['delta_mu']) <= 1:
            self.delta_mu = np.array(inp['delta_mu'], np.float64)
        else:
            self.delta_mu = np.linspace(
                inp['delta_mu'][0],
                inp['delta_mu'][1],
                inp['delta_mu'][2]
            )

        # Temperature
        if len(inp['temp']) == 1:
            raise NameError('must set temperature in input card')
        if len(inp['temp']) == 1:
            self.temp = np.array(inp['temp'], np.single)
        else:
            self.temp = np.linspace(
                inp['temp'][0],
                inp['temp'][1],
                inp['temp'][2]
            )

        # Boltzmann constant
        self.bzc = np.float32(inp['bzc'])

        # Interation energies
        self.int_pair = np.float64(inp['int_en']['pair'])
        self.int_trip = np.float64(inp['int_en']['trip'])
        self.int_tetra = np.float64(inp['int_en']['tetra'])

        # Concentration of impuity
        self.x_1 = np.float64(inp['x_1'])

        # Meta
        date_time_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.output['Meta']['Name'] = inp['name']
        self.output['Meta']['Description'] = inp['description']
        self.output['Meta']['Structure'] = inp['structure']
        self.output['Meta']['Date'] = date_time_str
        self.output['Meta']['Temperature'] = self.temp.tolist()
        self.output['Meta']['DeltaMu'] = self.delta_mu.tolist()

    def run(self):
        raise NameError('must implement this inherited abstract method')
