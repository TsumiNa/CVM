#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import datetime as dt
import threading

import numpy as np
from scipy.optimize import minimize_scalar

from .cluster_vibration import ClusterVibration as cv
from .unit_convert import *
from .sample import Sample


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
        'conv',  # unit conversion
        'series',  # calculation series
    )

    def __init__(self, inp):
        super(CVM, self).__init__()
        self.count = 0
        self.series = []
        self.output = {'meta': {}, 'results': [], 'experiment': {}}

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

        # coversion
        self.conv = np.float32(inp['conversion'])

        ##################
        # init series
        ##################
        if 'series' not in inp or len(inp['series']) == 0:
            raise NameError('need a defination of calculation series')

        for item in inp['series']:
            # sample holds all data for calculation
            sample = Sample(item['label'])
            self.series.append(sample)

            # initialzed impurity Concentration
            sample.x_1 = np.float64(item['x_1'])

            # convergence condition
            sample.condition = np.float32(item['condition'])

            # chemical potential
            if len(item['delta_mu']) <= 1:
                sample.mu = np.array(item['delta_mu'], np.float64)
            else:
                sample.mu = np.linspace(item['delta_mu'][0],
                                        item['delta_mu'][1],
                                        item['delta_mu'][2])

            # Temperature
            if len(item['temp']) == 0:
                raise NameError('must set temperature')
            if len(item['temp']) == 1:
                sample.temp = np.array(item['temp'], np.single)
            elif len(item['temp']) == 3:
                sample.temp = np.linspace(item['temp'][0], item['temp'][1],
                                          item['temp'][2])
            elif len(item['temp']) == 4:
                raise NameError('can not use this function right now')
            else:
                raise NameError('temperature was configured with error format')
            sample.res['temp'] = sample.temp

            # =================================================
            # Interation energies with thermal vibration
            # by combined with Morse potential and Debye model
            # ==================================================
            data = item['data']
            xs = np.array(data['lattice_c'])
            xs = lc2ad(xs)

            # Host with vibration
            # equilibrium lattice will evaluate from formula
            host = np.array(data['host_en']) * self.conv
            host_en = cv.free_energy(
                xs, host, 0, np.array(data['host_mass']), self.bzc)

            # get interaction energies
            def int_pair(r, T):
                # transter
                energies = []
                transfer = item['transfer']
                for n in ['pair' + str(s + 1) for s in np.arange(transfer[1])]:
                    _int = cv.int_energy(
                        xs, data[n], host, self.bzc, num=2, conv=self.conv)
                    energies.append(_int(r, T))
                energies[0] += np.float64(data['distortion'])
                return sample.effctive_en(energies, transfer)

            int_trip = cv.int_energy(
                xs, data['triple'], host, self.bzc, num=3, conv=self.conv)
            int_tetra = cv.int_energy(
                xs, data['tetra'], host, self.bzc, num=4, conv=self.conv)
            for T in np.nditer(sample.temp):
                r_0 = minimize_scalar(
                    lambda r: host_en(r, T),
                    bounds=(xs[0], xs[-1]), method='bounded'
                ).x
                pair = np.array(int_pair(r_0, T), np.float64)
                trip = np.array(int_trip(r_0, T), np.float64)
                tetra = np.array(int_tetra(r_0, T), np.float64)
                sample.int.append((pair, trip, tetra))

    def run(self):
        raise NotImplementedError(
            'must implement this inherited abstract method')
