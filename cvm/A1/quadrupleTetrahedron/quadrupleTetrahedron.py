#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from cvm.utilities import CVM

from .process import process


class quadrupleTetrahedron(CVM):

    """docstring for quadrupleTetrahedron"""

    __slots__ = (
        'm61_',  # 4body-1234, dim is 2x2x2x2x2x2
        'm51_',  # 4body-1234, dim is 2x2x2x2x2
        'm41_',  # 4body-1234, dim is 2x2x2x2
        'm42_',  # 4body-1234, dim is 2x2x2x2
        'm31_',  # 3body-126, dim is 2x2x2
        'm21_',  # pair-1st, dim is 2x2
        'm22_',  # pair-2nd, dim is 2x2
        'x_',  # point, dim is 2
        'qt_',  # 7-body, dim is 2x2x2x2x2x2x2x2x2
        'enQT',  # energy of tetrahedron, dim is 2x2x2x2x2x2x2x2x2
        'mu',  # opposite chemical potential, dim is 2
        'beta',  # β = 1/kt
        'checker',  # absolute of z_out and z_in
    )

    def __init__(self, inp):
        super(quadrupleTetrahedron, self).__init__(inp)
        ####################
        # define var
        ####################
        # double-tetrahedron
        self.qt_ = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2), np.float64)

        # 6-body
        self.m61_ = np.zeros((2, 2, 2, 2, 2, 2), np.float64)

        # 5-body
        self.m51_ = np.zeros((2, 2, 2, 2, 2), np.float64)

        # 4-body
        self.m41_ = np.zeros((2, 2, 2, 2), np.float64)

        # 4-body
        self.m42_ = np.zeros((2, 2, 2, 2), np.float64)

        # 3-body
        self.m31_ = np.zeros((2, 2, 2), np.float64)

        # pair
        self.m21_ = np.zeros((2, 2), np.float64)
        self.m22_ = np.zeros((2, 2), np.float64)

        # point
        self.x_ = np.zeros((2), np.float64)

        # energy
        self.enQT = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2), np.float64)

        # beta
        self.beta = np.float64(0.0)

        # mu
        self.mu = np.zeros((2), np.float64)

        ###############################################
        # configuration
        ###############################################
        # pure energy of 2body 1st ~ 4th
        e1 = np.zeros((2, 2), np.float64)
        e1[0, 1] = e1[1, 0] = 0.5 * (e1[0, 0] + e1[1, 1] - self.int_pair[0])

        # 2nd interaction energy
        de22 = np.zeros((2, 2), np.float64)
        de22[1, 1] = self.int_pair[1]

        # 3rd interaction energy
        de23 = np.zeros((2, 2), np.float64)
        de23[1, 1] = self.int_pair[2]

        # 3body-1st interaction energy
        de31 = np.zeros((2, 2, 2), np.float64)
        de31[1, 1, 1] = self.int_trip[0]

        # 4body-1st interaction energy
        de41 = np.zeros((2, 2, 2, 2), np.float64)
        de41[1, 1, 1, 1] = self.int_tetra[0]

        # # energy ε
        # it = np.nditer(self.enQT, flags=['multi_index'])
        # while not it.finished:
        #     i, j, k, l, m, n, o, p, q = it.multi_index
        #     self.enQT[i, j, k, l, m, n, o, p, q] = \
        #         (1 / 2) * (e1[i, j] + e1[i, k] + e1[i, l] + e1[j, k] +
        #                    e1[j, l] + e1[k, l] + e1[k, m] + e1[k, n] +
        #                    e1[l, m] + e1[l, n] + e1[m, n]) + \
        #         (1 / 4) * (de22[i, m] + de22[j, n]) + \
        #         (1 / 1) * (de23[i, n] + de23[j, m]) + \
        #         (1 / 3) * (de31[i, j, k] + de31[i, k, l] +
        #                    de31[k, l, i] + de31[k, l, j] +
        #                    de31[m, n, k] + de31[m, n, l] +
        #                    de31[k, l, n] + de31[k, l, m]) + \
        #         (1 / 3) * de41[i, j, k, l]
        #     it.iternext()

        # chemical potential
        self.mu[0] = (self.enQT[0, 0, 0, 0, 0, 0, 0, 0, 0] -
                      self.enQT[1, 1, 1, 1, 1, 1, 1, 1, 1])

    def __init(self):
        """
        initialize
        """
        self.count = 0
        self.checker = np.float64(1.0)

        it = np.nditer(self.qt_, flags=['multi_index'])
        while not it.finished:
            i, j, k, l, m, n, o, p, q = it.multi_index

            # qt_
            self.qt_[i, j, k, l, m, n, o, p, q] =\
                self.x_[i] * self.x_[j] * self.x_[k] * \
                self.x_[l] * self.x_[m] * self.x_[n] * \
                self.x_[o] * self.x_[p] * self.x_[q]

            # m61_
            self.m61_[q, o, j, n, i, k] += self.qt_[i, j, k, l, m, n, o, p, q]

            # m51_
            self.m51_[i, j, k, l, m] += self.qt_[i, j, k, l, m, n, o, p, q]

            # m41_
            self.m41_[i, j, n, p] += self.qt_[i, j, k, l, m, n, o, p, q]

            # m42_
            self.m42_[n, o, p, q] += self.qt_[i, j, k, l, m, n, o, p, q]

            # m31_
            self.m31_[i, k, j] += self.qt_[i, j, k, l, m, n, o, p, q]

            # m21_
            self.m21_[j, i] += self.qt_[i, j, k, l, m, n, o, p, q]

            # m22_
            self.m22_[i, k] += self.qt_[i, j, k, l, m, n, o, p, q]
            it.iternext()

    def __run(self):
        self.__init()
        while self.checker > self.condition:
            process(self)
        # process(self)

    # implement the inherited abstract method run()
    def run(self):

        # temperature iteration
        for dmu in np.nditer(self.delta_mu):
            data = []
            self.mu[0] += dmu
            self.mu[1] = -self.mu[0]
            self.x_[1] = self.x_1
            self.x_[0] = 1 - self.x_1
            print(' mu = {:06.4f}:'.format(self.mu[0].item(0)))
            print(' x1 = {:06.4f}:'.format(self.x_[1].item(0)))

            # delta mu iteration
            for temp in np.nditer(self.temp):
                self.beta = np.float64(pow(self.bzc * temp, -1))

                # calculate
                self.__run()

                # push result into data
                data.append({'temp': temp.item(0), 'c': self.x_[1].item(0)})
                print('    T = {:06.3f}K,  c = {:06.6f},  count = {}'.
                      format(temp.item(0), self.x_[1].item(0), self.count))

            print('\n')
            # save result to output
            self.output['Results'].append(
                {'mu': self.mu[0].item(0), 'data': data})
            self.mu[0] -= dmu
