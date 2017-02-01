#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .sample import Sample
from .function_tool import thermal_vibration_parameters
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
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

            # initialzed impurity Concentration
            sample.x_1 = np.float64(item['x_1'])

            # convergence condition
            sample.condition = np.float32(item['condition'])

            # =================================================
            # Interation energies with thermal vibration
            # by combined with Morse potential and Debye model
            # ==================================================
            data = item['data']
            xs = data['xdata']

            # gain free energy formula for host
            host_en = data['host']['energy']
            host_mass = data['host']['mass']
            vib_host = self.__energy_vib(xs, 0, host_en, 0, host_mass)
            
            # gain Interation energy for 1st-pair
            correct_en
            for cls in data['pair1']:


            # Equilibrium lattice constant
            for T in sample.temp:
                pass
            # transter
            if 'transfer' in item:
                sample.transfer = item['transfer']

            # Interation energies
            sample.int_pair = np.float64(item['int_pair'])
            sample.int_trip = np.float64(item['int_trip'])
            sample.int_tetra = np.float64(item['int_tetra'])

            # Concentration of impurity
            self.series.append(sample)

    def __energy_vib(self, xs, host, cluster, correct, mass):
        """
        xs: array
            atomic distance between nearest-neighbor
        correc, thost, cluster: array
             total energy of host
             energy different of impuity cluster
             correct between band calculation and impurity calculation
        """

        def __minimum(self, xs, ys, **kwagrs):
            """
            xs: array
                atomic distance between nearest-neighbor
            ys: array
                energies change for ys, impuity cluster and correct respectivly
            """
            base = 0 if 'base' not in kwagrs else kwagrs['base']
            correct = 0 if 'correct' not in kwagrs else kwagrs['correct']
            # get minimum from a polynomial
            ys = ys + base - correct
            poly_min = minimize_scalar(
                UnivariateSpline(xs, ys, k=4),
                bounds=(xs[0], xs[-1])
            )
            min_x = poly_min.x  # equilibrium atomic distance
            min_y = np.float(poly_min.fun)  # ground status energy

            return min_x, min_y

        # ys = cluster + host - correct
        # for example:
        # E_IIII = Imp_IIII(impuity calculation)
        #           + Host_HHHH(band calculation)
        #           - Correct_HHHH(impuity - band)

        # get polynomial minimum for morse fit
        _, min_y = __minimum(xs, cluster, base=host, correct=correct)

        # use polynomial minimum to obtain morse parameters
        ret = thermal_vibration_parameters(xs, cluster - min_y, mass)

        morse = ret['morse']  # Morse potential based on 0 minimum
        D = ret['debye_func']  # Debye function
        theta_D = ret['debye_temp_func']  # Debye Temperature function

        return lambda r, T: morse(r) + min_y \
            (9 / 8) * self.bzc * theta_D(r) - self.bzc * T * D(r, T) + \
            3 * self.bzc * T * np.log(1 - np.exp(-(theta_D(r) / T)))

    def run(self):
        raise NotImplementedError(
            'must implement this inherited abstract method')
