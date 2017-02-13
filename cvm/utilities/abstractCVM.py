#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import datetime as dt
import threading

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar

from .function_tool import thermal_vibration_parameters
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

        # host and imourity
        self.host_mass = inp['host_mass']
        self.impurity_mass = inp['impurity_mass']

        ##################
        # init series
        ##################
        if 'series' not in inp or len(inp['series']) == 0:
            raise NameError('need a defination of calculation series')

        for item in inp['series']:
            # sample holds all data for calculation
            sample = Sample(item['label'])
            self.series.append(sample)

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
            sample.res['temp'] = sample.temp.tolist()

            # initialzed impurity Concentration
            sample.x_1 = np.float64(item['x_1'])

            # convergence condition
            sample.condition = np.float32(item['condition'])

            # =================================================
            # Interation energies with thermal vibration
            # by combined with Morse potential and Debye model
            # ==================================================
            def __int_energy(num, xs, host, datas):
                """
                generate interaction energy
                """
                parts = list()
                for data in datas:
                    mass = data["mass"]
                    coeff = data["coefficient"]
                    energies = data["energy"]
                    part = self.__energy_vib(xs, host, energies / num,
                                             mass / num)
                    parts.append((coeff, part))

                def __int(r, T):
                    _int = np.float64(0)
                    for part in parts:
                        _int += part[0] * part[1](r, T)
                    return num * _int

                return __int

            data = item['data']
            xs = np.array(data['xdata'])

            # Host with vibration
            # equilibrium lattice will evaluate from formula
            host_en = np.array(data['host'])
            host_formula = self.__energy_vib(xs, 0, host_en, 0, self.host_mass)

            # Equilibrium lattice constant
            _temp = list()
            for T in np.nditer(sample.temp):
                r_0 = minimize_scalar(
                    lambda r: host_formula(r, T), bounds=(xs[0], xs[-1]))
                _temp.append(r_0, T)
            sample.temp = np.array(_temp)

            # transter
            if 'transfer' in item:
                sample.transfer = item['transfer']

            # Interation energies
            sample.int_pari_1 = __int_energy(2, xs, host_en, data["pair1"])
            sample.int_trip = __int_energy(3, xs, host_en, data["triple"])
            sample.int_tetra = __int_energy(4, xs, host_en, data["tetra"])

    def __minimum(self, xs, ys, **kwagrs):
        """
        xs: array
            atomic distance between nearest-neighbor
        ys: array
            energies change for ys, impuity cluster and correct respectivly
        """

        # get minimum from a polynomial
        poly_min = minimize_scalar(
            UnivariateSpline(xs, ys, k=4), bounds=(xs[0], xs[-1]))
        min_x = poly_min.x  # equilibrium atomic distance
        min_y = np.float(poly_min.fun)  # ground status energy

        return min_x, min_y

    def __energy_vib(self, xs, host, cluster, correct, mass):
        """
        xs: array
            atomic distance between nearest-neighbor
        correc, thost, cluster: array
             total energy of host
             energy different of impuity cluster
             correct between band calculation and impurity calculation
        """

        # ys = cluster + host - correct
        # for example:
        # E_IIII = Imp_IIII(impuity calculation)
        #           + Host_HHHH(band calculation)
        #           - Correct_HHHH(impuity - band)

        # get polynomial minimum for morse fit
        _, min_y = self.__minimum(xs, cluster + host - correct)

        # use polynomial minimum to obtain morse parameters
        ret = thermal_vibration_parameters(xs, cluster - min_y, mass)

        morse = ret['morse']  # Morse potential based on 0 minimum
        D = ret['debye_func']  # Debye function
        theta_D = ret['debye_temp_func']  # Debye Temperature function

        return lambda r, T: morse(r) + min_y + \
            (9 / 8) * self.bzc * theta_D(r) - self.bzc * T * D(r, T) + \
            3 * self.bzc * T * np.log(1 - np.exp(-(theta_D(r) / T)))

    def run(self):
        raise NotImplementedError(
            'must implement this inherited abstract method')
