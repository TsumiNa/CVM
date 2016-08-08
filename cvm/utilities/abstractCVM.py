#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
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
        'condition',  # convergence condition
        'delta_mu',  # opposite chemical potential
        'int_pair',  # pair interaction energy
        'int_trip',  # triple interaction energy
        'int_tetra'  # tetrahedron interaction energy
    )

    def __init__(self, inp):
        super(CVM, self).__init__()
        self.count = 0
        self.output = {'Meta': {}, 'Results': []}

        ##################
        # check input
        ##################
        if 'int_en' not in inp:
            print('Need interaction energy')
            sys.exit(0)

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
        if len(inp['temp']) == 0:
            raise NameError('must set temperature in input card')
        if len(inp['temp']) == 1:
            self.temp = np.array(inp['temp'], np.single)
        elif len(inp['temp']) == 3:
            self.temp = np.linspace(
                inp['temp'][0],
                inp['temp'][1],
                inp['temp'][2]
            )
        elif len(inp['temp']) == 4:
            pass
        else:
            print(len(inp['temp']))
            raise NameError('wrong temperature set')

        # Boltzmann constant
        self.bzc = np.float32(inp['bzc'])

        # convergence condition
        self.condition = np.float32(inp['condition'])

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
        self.output['Meta']['Experiment'] = inp['experiment']

    def run(self):
        raise NameError('must implement this inherited abstract method')

    def pair_int_decorator(self, transfer_to=1, cut=0):
        """
        2nd parameter refer to the neighbour that transfer to
        """
        # coordination number
        _coord_num = np.array([12,      # 1st
                               6,       # 2nd
                               24,      # 3rd
                               12,      # 4th
                               24,      # 5th
                               8,       # 6th
                               48,      # 7th
                               6,       # 8th
                               24,      # 9th
                               12,      # 10th
                               24,      # 11th
                               24,      # 12th
                               48,      # 13th
                               48,      # 14th
                               12,      # 15th
                               24,      # 16th
                               24,      # 17th
                               6,       # 17th
                               48,      # 18th
                               24,      # 19th
                               48])     # 20th

        # calculation pair interaction
        _int = np.float64(0.0)
        _range = range(transfer_to - 1, len(self.int_pair) - cut)
        for index in _range:
            _int += _coord_num[index] * self.int_pair[index] /\
                _coord_num[transfer_to - 1]
            print('pair interaction with %sth approximation is %s' %
                  (index + 1, _int))
        self.int_pair[transfer_to - 1] = _int
