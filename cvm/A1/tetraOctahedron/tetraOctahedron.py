#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from cvm.utilities import CVM

from .process import process


class tetraOctahedron(CVM):

    """docstring for tetraOctahedron"""

    __slots__ = (
        'x_',  # point, dim is 2
        'y_',  # pair, dim is 2x2
        'z_',  # triangle, dim is 2x2x2
        'zt_',  # triangle from T, dim is 2x2x2
        'zo_',  # triangle from O, dim is 2x2x2
        'af_',  # perturbation from octahedron, dim is 2x2x2
        'enT',  # energy of tetrahedron, dim is 2x2x2x2
        'enO',  # energy of octahedron, dim is 2x2x2x2x2x2
        'mu',  # opposite chemical potential, dim is 2
        'beta',  # β = 1/kt
        'checker',  # absolute of z_out and z_in
        'main_condition',  # Convergence condition
        'sub_condition',  # Convergence condition
    )

    def __init__(self, inp):
        super(tetraOctahedron, self).__init__(inp)
        ####################
        # define var
        ####################
        self.x_ = np.zeros((2), np.float64)
        self.y_ = np.zeros((2, 2), np.float64)
        self.z_ = np.zeros((2, 2, 2), np.float64)
        self.enT = np.zeros((2, 2, 2, 2), np.float64)
        self.enO = np.zeros((2, 2, 2, 2, 2, 2), np.float64)
        self.beta = np.float64(0.0)
        self.mu = np.zeros((2), np.float64)

        ###############################################
        # configuration
        ###############################################
        # pure energy of 2body-1st
        en = np.zeros((2, 2), np.float64)
        en[0, 1] = en[1, 0] = 0.5 * (en[0, 0] + en[1, 1] - self.int_pair[0])

        #############################################
        # tetrahedron
        #############################################
        # 3body-1st interaction energy
        de31 = np.zeros((2, 2, 2), np.float64)
        de31[1, 1, 1] = self.int_trip[0]

        # 4body-1st interaction energy
        de41 = np.zeros((2, 2, 2, 2), np.float64)
        de41[1, 1, 1, 1] = self.int_tetra[0]

        # energy ε
        it = np.nditer(self.enT, flags=['multi_index'])
        while not it.finished:
            i, j, k, l = it.multi_index
            self.enT[i, j, k, l] = \
                0.5 * (en[i, j] + en[i, k] + en[i, l] +
                       en[j, k] + en[j, l] + en[l, k]) +\
                (de31[i, j, k] + de31[i, k, l] +
                 de31[i, j, l] + de31[j, k, l]) +\
                de41[i, j, k, l]
            it.iternext()
        # =============================================

        ########################
        # octahedron
        ########################
        # pure energy of 2body-1st
        en2 = np.zeros((2, 2), np.float64)
        en2[0, 1] = en2[1, 0] = \
            0.5 * (en2[0, 0] + en2[1, 1] - self.int_pair[1])

        # energy ε
        it = np.nditer(self.enO, flags=['multi_index'])
        while not it.finished:
            i, j, k, l, m, n = it.multi_index
            self.enO[i, j, k, l, m, n] = en2[i, k] + en2[j, l] + en2[n, m]
            it.iternext()
        # ==============================================

        # chemical potential
        self.mu[0] = (self.enT[0, 0, 0, 0] + self.enO[0, 0, 0, 0, 0, 0]) - \
            (self.enT[1, 1, 1, 1] + self.enO[1, 1, 1, 1, 1, 1])
        # print('mu is: {}'.format(self.mu[0]))

    def __init(self):
        """
        initialize x_, y_, z_
        """
        self.count = 0
        self.checker = np.float64(1.0)
        self.af_ = np.zeros((2, 2, 2), np.float64)
        self.main_condition = np.float64(1e-3)
        self.sub_condition = np.float64(1e-1)

        it = np.nditer(self.z_, flags=['multi_index'])
        while not it.finished:
            i, j, k = it.multi_index
            self.z_[i, j, k] = self.x_[i] * self.x_[j] * self.x_[k]
            self.y_[i, j] = self.x_[i] * self.x_[j]
            it.iternext()

    def __run(self):
        self.__init()
        while self.checker > self.condition:
            while self.checker > self.main_condition:
                # print('process run')
                process(self)
            else:
                self.main_condition /= 10
                self.sub_condition /= 10

    # implement the inherited abstract method run()
    def run(self):

        # temperature iteration
        for dmu in np.nditer(self.delta_mu):
            data = []
            self.mu[0] += dmu
            self.mu[1] = -self.mu[0]
            self.x_[0] = self.x_1
            self.x_[1] = 1 - self.x_1
            print(' mu = {:06.4f}:'.format(self.mu[0].item(0)))

            # delta mu iteration
            for temp in np.nditer(self.temp):
                self.beta = np.float64(pow(self.bzc * temp, -1))

                # calculate w
                self.__run()

                # push result into data
                data.append({'temp': temp.item(0), 'c': self.x_[0].item(0)})
                print('    T = {:06.3f}K,  c = {:06.6f},  conut = {}'.
                      format(temp.item(0), self.x_[0].item(0), self.count))

            print('\n')
            # save result to output
            self.output['Results'].append(
                {'mu': self.mu[0].item(0), 'data': data})
            self.mu[0] -= dmu