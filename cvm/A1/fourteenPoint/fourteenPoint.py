#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from cvm.utilities import CVM

from .process import process


class fourteenPoint(CVM):

    """docstring for fourteenPoint"""

    __slots__ = (
        'm61_',  # DT, dim is 2x2x2x2x2x2
        'm51_',  # Square, dim is 2x2x2x2x2
        'm41_',  # Square, dim is 2x2x2x2
        'm31_',  # Square, dim is 2x2x2
        'm21_',  # pair-1st, dim is 2x2
        'x_',  # point, dim is 2
        'fp_',  # 7-body, dim is 2x2x2x2x2x2x2x2x2x2x2x2x2x2
        'enFP',  # energy, dim is 2x2x2x2x2x2x2x2x2x2x2x2x2x2
        'mu',  # opposite chemical potential, dim is 2
        'beta',  # β = 1/kt
        'checker',  # absolute of z_out and z_in
    )

    def __init__(self, inp):
        super(fourteenPoint, self).__init__(inp)
        ####################
        # define var
        ####################
        # double-tetrahedron
        self.fp_ = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
                            np.float64)

        # DT
        self.m61_ = np.zeros((2, 2, 2, 2, 2, 2), np.float64)

        # Square
        self.m51_ = np.zeros((2, 2, 2, 2, 2), np.float64)

        # pair
        self.m41_ = np.zeros((2, 2, 2, 2), np.float64)

        # 3-body
        self.m31_ = np.zeros((2, 2, 2), np.float64)

        # 1st-pair
        self.m21_ = np.zeros((2, 2), np.float64)

        # point
        self.x_ = np.zeros((2), np.float64)

        # energy
        self.enFP = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
                             np.float64)

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

        # energy ε
        # it = np.nditer(self.enFP, flags=['multi_index'])
        # while not it.finished:
        #     i, j, k, l, m, n, o, p, q, r, s, t, u, v = \
        #         it.multi_index
        #     self.enFP[i, j, k, l, m, n, o, p, q, r, s, t, u, v] = \
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
        self.mu[0] = (self.enFP[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] -
                      self.enFP[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def __init(self):
        """
        initialize
        """
        self.count = 0
        self.checker = np.float64(1.0)

        it = np.nditer(self.m61_, flags=['multi_index'])
        while not it.finished:
            i, j, k, l, m, n = it.multi_index

            # m61_
            self.m61_[i, j, k, l, m, n] = \
                self.x_[i] * self.x_[j] * self.x_[k] * \
                self.x_[l] * self.x_[m] * self.x_[n]

            # m51_
            self.m51_[i, j, k, l, m] = \
                self.x_[i] * self.x_[j] * self.x_[k] * \
                self.x_[l] * self.x_[m]

            # m41_
            self.m41_[i, j, k, l] = \
                self.x_[i] * self.x_[j] * \
                self.x_[l] * self.x_[m]

            # m31_
            self.m31_[i, j, k] = \
                self.x_[i] * self.x_[j] * self.x_[k]

            # m21_
            self.m21_[i, j] = self.x_[i] * self.x_[j]

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
