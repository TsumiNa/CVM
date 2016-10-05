#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .sample import Sample
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
        'bzc',  # Boltzmann constant
        'series',  # calculation series
    )

    def __init__(self, inp):
        super(CVM, self).__init__()
        self.count = 0
        self.series = []
        self.output = {
            'Meta': {},
            'Results': [],
            'Experiment': {}
        }

        ##################
        # init output
        ##################
        # experiment
        self.output['Experiment'] = inp['experiment']

        # Meta
        date_time_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.output['Meta']['Name'] = inp['name']
        self.output['Meta']['Description'] = inp['description']
        self.output['Meta']['Structure'] = inp['structure']
        self.output['Meta']['Date'] = date_time_str

        # Boltzmann constant
        self.bzc = np.float32(inp['bzc'])

        ##################
        # init series
        ##################
        if 'series' not in inp or len(inp['series']) == 0:
            raise NameError('need a defination of calculation series')

        for item in inp['series']:
            sample = Sample(item['label'])
            # chemical potential
            if len(item['delta_mu']) <= 1:
                sample.mu = np.array(item['delta_mu'], np.float64)
            else:
                sample.mu = np.linspace(
                    item['delta_mu'][0],
                    item['delta_mu'][1],
                    item['delta_mu'][2]
                )

            # Temperature
            if len(item['temp']) == 0:
                raise NameError('must set temperature')
            if len(item['temp']) == 1:
                sample.temp = np.array(item['temp'], np.single)
            elif len(item['temp']) == 3:
                sample.temp = np.linspace(
                    item['temp'][0],
                    item['temp'][1],
                    item['temp'][2]
                )
            elif len(item['temp']) == 4:
                raise NameError('can not use this function right now')
            else:
                raise NameError('temperature was configured with error format')

            # convergence condition
            sample.condition = np.float32(item['condition'])

            # Interation energies
            sample.int_pair = np.float64(item['int_pair'])
            sample.int_trip = np.float64(item['int_trip'])
            sample.int_tetra = np.float64(item['int_tetra'])

            # Concentration of impuity
            sample.x_1 = np.float64(item['x_1'])

            self.series.append(sample)

    def run(self):
        raise NameError('must implement this inherited abstract method')
