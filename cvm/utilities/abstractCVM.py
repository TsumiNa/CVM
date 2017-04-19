#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .sample import Sample
import numpy as np
import threading
import datetime as dt


class CVM(threading.Thread):

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
            'meta': {},
            'results': [],
            'experiment': {}
        }

        ##################
        # init output
        ##################
        # experiment
        self.output['experiment'] = inp['experiment']

        # Meta
        date_time_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.output['meta']['host'] = inp['host'].lower()
        self.output['meta']['impurity'] = inp['impurity'].lower()
        self.output['meta']['suffix'] = inp['suffix'].lower()
        self.output['meta']['prefix'] = inp['prefix'].lower()
        self.output['meta']['description'] = inp['description']
        self.output['meta']['structure'] = inp['structure']
        self.output['meta']['date'] = date_time_str

        # Boltzmann constant
        self.bzc = np.float32(inp['bzc'])

        ##################
        # init series
        ##################
        if 'series' not in inp or len(inp['series']) == 0:
            raise NameError('need a defination of calculation series')

        for item in inp['series']:
            # sample holds all data for calculation
            sample = Sample(
                item['label'],
                [
                    12,
                    6,
                    24,
                    12,
                    24,
                    8,
                    48,
                    6,
                    12,  # 9th-a
                    24,  # 9th-b
                    4,
                    24,
                    24,
                    48,  # 13th-a
                    24,  # 13th-b
                    48,
                    12,
                    24,  # 16th-a
                    24,  # 16th-b
                    24,  # 17th-a
                    6,   # 17th-b
                    48,  # 18th-a
                    24,  # 18th-b
                    24,
                    48  # 20th
                ])

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

            # transter
            en_pairs = np.float64(item['int_pair'])
            int_pair = []
            if 'transfer' in item:
                transfer = item['transfer']
                int_pair = sample.effctive_en(en_pairs, *transfer)
            else:
                int_pair = en_pairs

            # Interation energies
            sample.int_pair = int_pair
            sample.int_trip = np.float64(item['int_trip'])
            sample.int_tetra = np.float64(item['int_tetra'])

            # Concentration of impuity
            sample.x_1 = np.float64(item['x_1'])

            self.series.append(sample)

    def run(self):
        raise NotImplementedError('must implement this inherited abstract method')
