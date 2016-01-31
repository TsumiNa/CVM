#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from cvm.utilities import CVM

from .process import process


class quadrupleTetrahedron(CVM):

    """docstring for quadrupleTetrahedron"""

    __slots__ = (
        'm61_',  # 6body-DT, dim is 2x2x2x2x2x2
        'm51_',  # 5body-Square, dim is 2x2x2x2x2
        'm41_',  # 4body-T, dim is 2x2x2x2
        'm42_',  # 4body-1st-Square, dim is 2x2x2x2
        'm311_',  # 3body-ijk, dim is 2x2x2
        'm312_',  # 3body-ink, dim is 2x2x2
        'm313_',  # 3body-njp, dim is 2x2x2
        'm314_',  # 3body-qjo, dim is 2x2x2
        'm211_',  # pair-1st, dim is 2x2
        'm212_',  # pair-1st, dim is 2x2
        'm213_',  # pair-1st, dim is 2x2
        'm214_',  # pair-1st, dim is 2x2
        'm215_',  # pair-1st, dim is 2x2
        'm221_',  # pair-ik, dim is [2x2]^2
        'm222_',  # pair-np, dim is [2x2]^2
        'x1_',  # point, dim is 2
        'x2_',  # point, dim is 2
        'x3_',  # point, dim is 2
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
        self.m311_ = np.zeros((2, 2, 2), np.float64)
        self.m312_ = np.zeros((2, 2, 2), np.float64)
        self.m313_ = np.zeros((2, 2, 2), np.float64)
        self.m314_ = np.zeros((2, 2, 2), np.float64)

        # pair-1st
        self.m211_ = np.zeros((2, 2), np.float64)
        self.m212_ = np.zeros((2, 2), np.float64)
        self.m213_ = np.zeros((2, 2), np.float64)
        self.m214_ = np.zeros((2, 2), np.float64)
        self.m215_ = np.zeros((2, 2), np.float64)

        # pair-2nd
        self.m221_ = np.zeros((2, 2), np.float64)
        self.m222_ = np.zeros((2, 2), np.float64)

        # point
        self.x1_ = np.zeros((2), np.float64)
        self.x2_ = np.zeros((2), np.float64)
        self.x3_ = np.zeros((2), np.float64)

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
                self.x1_[i] * self.x1_[j] * self.x1_[k] * \
                self.x1_[l] * self.x1_[m] * self.x1_[n] * \
                self.x1_[o] * self.x1_[p] * self.x1_[q]

            # m61_
            self.m61_[i, j, k, l, m, n] += self.qt_[i, j, k, l, m, n, o, p, q]

            # m51_
            self.m51_[i, j, m, o, q] += self.qt_[i, j, k, l, m, n, o, p, q]

            # m41_
            self.m41_[i, j, k, l] += self.qt_[i, j, k, l, m, n, o, p, q]

            # m42_
            self.m42_[k, n, p, l] += self.qt_[i, j, k, l, m, n, o, p, q]

            # m31_
            self.m311_[i, j, m] += self.qt_[i, j, k, l, m, n, o, p, q]
            self.m312_[i, k, m] += self.qt_[i, j, k, l, m, n, o, p, q]
            self.m313_[k, j, p] += self.qt_[i, j, k, l, m, n, o, p, q]
            self.m314_[k, n, p] += self.qt_[i, j, k, l, m, n, o, p, q]

            # m21_
            self.m211_[i, j] += self.qt_[i, j, k, l, m, n, o, p, q]
            self.m212_[i, k] += self.qt_[i, j, k, l, m, n, o, p, q]
            self.m213_[i, l] += self.qt_[i, j, k, l, m, n, o, p, q]
            self.m214_[k, j] += self.qt_[i, j, k, l, m, n, o, p, q]
            self.m215_[l, k] += self.qt_[i, j, k, l, m, n, o, p, q]

            # m22_
            self.m221_[i, m] += self.qt_[i, j, k, l, m, n, o, p, q]
            self.m222_[k, p] += self.qt_[i, j, k, l, m, n, o, p, q]

            it.iternext()

    def __run(self):
        self.__init()
        # while self.checker > self.condition:
            # process(self)
        process(self)

    # implement the inherited abstract method run()
    def run(self):

        # temperature iteration
        for dmu in np.nditer(self.delta_mu):
            data = []
            self.mu[0] += dmu
            self.mu[1] = -self.mu[0]
            self.x1_[1] = self.x_1
            self.x1_[0] = 1 - self.x_1
            self.x3_ = self.x2_ = self.x1_
            print(' mu = {:06.4f}:'.format(self.mu[0].item(0)))
            print(' x1 = {:06.4f}:'.format(self.x1_[1].item(0)))

            # delta mu iteration
            for temp in np.nditer(self.temp):
                self.beta = np.float64(pow(self.bzc * temp, -1))

                # calculate
                self.__run()

                # push result into data
                data.append({'temp': temp.item(0), 'c': self.x1_[1].item(0)})
                print('    T = {:06.3f}K,  c = {:06.6f},  count = {}'.
                      format(temp.item(0), self.x1_[1].item(0), self.count))

            print('\n')
            # save result to output
            self.output['Results'].append(
                {'mu': self.mu[0].item(0), 'data': data})
            self.mu[0] -= dmu
