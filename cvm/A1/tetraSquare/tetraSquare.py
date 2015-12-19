#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from cvm.utilities import CVM

from .process import process


class tetraSquare(CVM):

    """docstring for tetraSquare"""

    __slots__ = (
        'x_',  # point, dim is 2
        'm21_',  # pair-1st, dim is 2x2
        'm22_',  # pair-2nd, dim is 2x2
        'm23_',  # pair-1st, dim is 2x2
        'm24_',  # pair-2nd, dim is 2x2
        'm31_',  # pair-2nd, dim is 2x2x2
        'm32_',  # pair-2nd, dim is 2x2x2
        'm34_',  # pair-2nd, dim is 2x2x2
        'm41_',  # 4body-1234, dim is 2x2x2x2
        'm42_',  # 4body-1234, dim is 2x2x2x2
        'm45_',  # 4body-1234, dim is 2x2x2x2
        'm46_',  # 4body-1234, dim is 2x2x2x2
        'm51_',  # 5body-12567, dim is 2x2x2x2x2
        'm52_',  # 5body-12345, dim is 2x2x2x2x2
        'ts_',  # 7-body, dim is 2x2x2x2x2x2x2
        'enTS',  # energy of tetrahedron, dim is 2x2x2x2
        'mu',  # opposite chemical potential, dim is 2
        'beta',  # β = 1/kt
        'checker',  # absolute of z_out and z_in
    )

    def __init__(self, inp):
        super(tetraSquare, self).__init__(inp)
        ####################
        # define var
        ####################
        self.x_ = np.zeros((2), np.float64)
        self.m21_ = np.zeros((2, 2), np.float64)
        self.m22_ = np.zeros((2, 2), np.float64)
        self.m23_ = np.zeros((2, 2), np.float64)
        self.m24_ = np.zeros((2, 2), np.float64)
        self.m31_ = np.zeros((2, 2, 2), np.float64)
        self.m32_ = np.zeros((2, 2, 2), np.float64)
        self.m34_ = np.zeros((2, 2, 2), np.float64)
        self.m41_ = np.zeros((2, 2, 2, 2), np.float64)
        self.m42_ = np.zeros((2, 2, 2, 2), np.float64)
        self.m45_ = np.zeros((2, 2, 2, 2), np.float64)
        self.m46_ = np.zeros((2, 2, 2, 2), np.float64)
        self.m51_ = np.zeros((2, 2, 2, 2, 2), np.float64)
        self.m52_ = np.zeros((2, 2, 2, 2, 2), np.float64)
        self.ts_ = np.zeros((2, 2, 2, 2, 2, 2, 2), np.float64)
        self.enTS = np.zeros((2, 2, 2, 2, 2, 2, 2), np.float64)
        self.beta = np.float64(0.0)
        self.mu = np.zeros((2), np.float64)

        ###############################################
        # configuration
        ###############################################
        # pure energy of 2body 1st ~ 4th
        e1 = np.zeros((2, 2), np.float64)
        e1[0, 1] = e1[1, 0] = 0.5 * (e1[0, 0] + e1[1, 1] - self.int_pair[0])

        e2 = np.zeros((2, 2), np.float64)
        e2[0, 1] = e2[1, 0] = 0.5 * (e2[0, 0] + e2[1, 1] - self.int_pair[1])

        e3 = np.zeros((2, 2), np.float64)
        e3[0, 1] = e3[1, 0] = 0.5 * (e3[0, 0] + e3[1, 1] - self.int_pair[2])

        e4 = np.zeros((2, 2), np.float64)
        e4[0, 1] = e4[1, 0] = 0.5 * (e4[0, 0] + e4[1, 1] - self.int_pair[3])

        # 2nd interaction energy
        de22 = np.zeros((2, 2), np.float64)
        de22[1, 1] = self.int_pair[1]

        # 3rd interaction energy
        de23 = np.zeros((2, 2), np.float64)
        de23[1, 1] = self.int_pair[2]

        # 4th interaction energy
        de24 = np.zeros((2, 2), np.float64)
        de24[1, 1] = self.int_pair[3]

        # 3body-1st interaction energy
        de31 = np.zeros((2, 2, 2), np.float64)
        de31[1, 1, 1] = self.int_trip[0]

        # 4body-1st interaction energy
        de41 = np.zeros((2, 2, 2, 2), np.float64)
        de41[1, 1, 1, 1] = self.int_tetra[0]

        # energy ε
        it = np.nditer(self.enTS, flags=['multi_index'])
        # while not it.finished:
        #     i, j, k, l, m, n, o = it.multi_index
        #     self.enTS[i, j, k, l, m, n, o] = \
        #         (1 / 44) * (e1[i, j] + e1[i, k] + e1[i, l] + e1[j, k] +
        #                     e1[j, l] + e1[j, m] + e1[j, n] + e1[j, o] +
        #                     e1[o, l] + e1[l, k] + e1[k, m]) +\
        #         (1 / 32) * (e2[i, m] + e2[i, o] + e2[n, m] + e2[n, o]) +\
        #         (1 / 32) * (e3[k, n] + e3[k, o] + e3[l, n] + e3[l, m]) +\
        #         (1 / 16) * (e4[i, n] + e4[m, o]) +\
        #         (de31[i, j, k] + de31[i, k, l] + de31[i, j, l] +
        #          de31[j, k, l] + de31[j, k, m] + de31[j, l, o]) +\
        #         de41[i, j, k, l]
        #     # print('self.enTS{} is: {}'.format(it.multi_index, self.enTS[i, j, k, l, m, n, o]))
        #     it.iternext()
        while not it.finished:
            i, j, k, l, m, n, o = it.multi_index
            self.enTS[i, j, k, l, m, n, o] = \
                (1 / 44) * (e1[i, j] + e1[i, k] + e1[i, l] + e1[j, k] +
                            e1[j, l] + e1[j, m] + e1[j, n] + e1[j, o] +
                            e1[o, l] + e1[l, k] + e1[k, m]) +\
                (1 / 32) * (de22[i, m] + de22[i, o] + de22[n, m] + de22[n, o]) +\
                (1 / 32) * (de23[k, n] + de23[k, o] + de23[l, n] + de23[l, m]) +\
                (1 / 16) * (de24[i, n] + de24[m, o]) +\
                (1 / 8) * (de31[i, j, k] + de31[i, k, l] + de31[i, j, l] +
                           de31[j, k, l] + de31[j, k, m] + de31[j, l, o]) +\
                (1 / 4) * de41[i, j, k, l]
            # print('self.enTS{} is: {}'.format(it.multi_index, self.enTS[i, j, k, l, m, n, o]))
            it.iternext()
        # =============================================

        # chemical potential
        self.mu[0] = (self.enTS[0, 0, 0, 0, 0, 0, 0] -
                      self.enTS[1, 1, 1, 1, 1, 1, 1])
        # print('mu is: {}'.format(self.mu[0]))

    def __init(self):
        """
        initialize
        """
        self.count = 0
        self.checker = np.float64(1.0)

        it = np.nditer(self.m51_, flags=['multi_index'])
        while not it.finished:
            i, j, k, l, m = it.multi_index
            self.m51_[i, j, k, l, m] =\
                (1 / 9) * (self.x_[i] * self.x_[j] * self.x_[k] *
                           self.x_[l] * self.x_[m])
            self.m52_[i, j, k, l, m] =\
                (8 / 9) * (self.x_[i] * self.x_[j] * self.x_[k] *
                           self.x_[l] * self.x_[m])
            self.m4_[i, j, k, l] =\
                self.x_[i] * self.x_[j] * self.x_[k] * self.x_[l]
            self.m21_[i, j] = (2 / 3) * (self.x_[i] * self.x_[j])
            self.m22_[i, j] = (1 / 3) * (self.x_[i] * self.x_[j])
            it.iternext()

    def __run(self):
        self.__init()
        while self.checker > self.condition:
            process(self)

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
                print('    T = {:06.3f}K,  c = {:06.6f},  conut = {}'.
                      format(temp.item(0), self.x_[1].item(0), self.count))

            print('\n')
            # save result to output
            self.output['Results'].append(
                {'mu': self.mu[0].item(0), 'data': data})
            self.mu[0] -= dmu
