#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from cvm.utilities import CVM

from .octahedron import wO
from .tetrahedron import wT

class tetraOctahedron(CVM):

    """docstring for tetraOctahedron"""

    __slots__ = (
        'x_',  # concentration of point, dim is 2
        'y_',  # concentration of pair, dim is 2x2
        'z_',  # concentration of triangle, dim is 2x2x2
        'wt_',  # concentration of tetrahedron, dim is 2x2x2x2
        'wo_',  # concentration of tetrahedron, dim is 2x2x2x2x2x2
        'enT',  # energy of tetrahedron, dim is 2x2x2x2
        'enO',  # energy of octahedron, dim is 2x2x2x2x2x2
        'mu',  # opposite chemical potential, dim is 2
        'beta',  # β = 1/kt
        'eta_sum',  # sum of η_ijkl
    )

    def __init__(self, inp):
        super(tetraOctahedron, self).__init__(inp)
        ####################
        # define var
        ####################
        self.x_ = np.zeros((2), np.float64)
        self.y_ = np.zeros((2, 2), np.float64)
        self.z_ = np.zeros((2, 2, 2), np.float64)
        self.wt_ = np.zeros((2, 2, 2, 2), np.float64)
        self.wo_ = np.zeros((2, 2, 2, 2), np.float64)
        self.enT = np.zeros((2, 2, 2, 2), np.float64)
        self.enO = np.zeros((2, 2, 2, 2, 2, 2), np.float64)
        self.beta = np.float64(0.0)
        self.mu = np.zeros((2), np.float64)
        self.eta_sum = np.float64(0.0)

        # init
        self.__init()

    def __init(self):
        self.x_[0] = self.x_1
        self.x_[1] = 1 - self.x_1

        self.y_[0, 0] = self.x_[0]**2
        self.y_[0, 1] = self.y_[1, 0] = self.x_[0] * self.x_[1]
        self.y_[1, 1] = self.x_[1]**2

        # pure energy of 2body-1st
        en = np.zeros((2, 2), np.float64)
        en[0, 1] = en[0, 1] = 0.5 * (en[0, 0] + en[1, 1] - self.int_pair[0])
        # en2 = np.zeros((2, 2), np.float64)
        # en2[0, 1] = en2[0, 1] = \
        #     0.5 * (en2[0, 0] + en2[1, 1] - self.int_pair[1])

        # ε^tetrahedron
        de31 = np.zeros((2, 2, 2), np.float64)  # 3body-1st interaction energy
        de31[1, 1, 1] = self.int_trip[0]
        de41 = np.zeros((2, 2, 2), np.float64)  # 4body-1st interaction energy
        de41[1, 1, 1, 1] = self.int_tetra[0]
        it = np.nditer(self.enT, flags=['multi_index'])
        while not it.finished:
            i, j, k, l = it.multi_index
            self.enT[i, j, k, l] = \
                0.5 * (en[i, j] + en[i, k] + en[i, l] +
                       en[j, k] + en[l, j] + en[k, l]) +\
                (de31[i, j, k] + de31[i, k, l] +
                 de31[i, j, l] + de31[j, k, l]) +\
                de41[i, j, k, l]

        # ε^octahedron
        de22 = np.zeros((2, 2), np.float64)  # 2body-2nd interaction energy
        de22[1, 1] = self.int_pair[1]
        it = np.nditer(self.enO, flags=['multi_index'])
        while not it.finished:
            i, j, k, l, m, n = it.multi_index
            self.enO[i, j, k, l, m, n] = \
                0.5 * (en[i, j] + en[i, l] + en[i, m] + en[i, n] +
                       en[j, k] + en[j, m] + en[j, n] + en[k, l] +
                       en[k, m] + en[k, n] + en[l, m] + en[l, n]) +\
                (de22[i, k] + de22[j, l] + de22[n, m])  # interaction energy

        self.mu[0] = self.enT[0, 0, 0, 0] + self.enO[0, 0, 0, 0, 0, 0] - \
            self.enT[1, 1, 1, 1] - self.enO[1, 1, 1, 1, 1, 1]
        self.mu[1] = -self.mu[0]

    # implement the inherited abstract method run()
    def run(self):

        # temperature iteration
        for dmu in np.nditer(self.delta_mu):
            data = []
            self.mu[0] += dmu
            # print('mu = {:06.4f}:'.format(self.mu[0].item(0)))

            # delta mu iteration
            for temp in np.nditer(self.temp):
                self.beta = np.float64(pow(self.bzc * temp, -1))
                self.x_[0] = self.x_1
                self.x_[1] = 1 - self.x_1

                self.y_[0, 0] = self.x_[0]**2
                self.y_[0, 1] = self.y_[1, 0] = self.x_[0] * self.x_[1]
                self.y_[1, 1] = self.x_[1]**2

                self.__run()
                data.append(
                    {'temp': temp.item(0), 'c': self.x_[0].item(0)})
                # print('    T = {:06.4f}K,  c = {:06.4f}'.format(
                #     temp.item(0), self.x_[0].item(0)))

            # print('\n')
            # save result to output
            self.output['Results'].append(
                {'mu': self.mu[0].item(0), 'data': data})
            self.mu[0] -= dmu

    def __run(self):
        """
        TODO: need doc
        """
        eta_sum = self.eta_sum

        # calculate z_ijkl
        self.__z_ijkl()

        # counts
        self.count += 1

        # y_
        self.y_[0, 0] = 1 * self.wt_[0, 0, 0, 0] + \
            2 * self.wt_[1, 0, 0, 0] + \
            1 * self.wt_[1, 1, 0, 0]
        self.y_[1, 0] = self.y_[0, 1] = 1 * self.wt_[1, 0, 0, 0] + \
            2 * self.wt_[1, 1, 0, 0] + \
            1 * self.wt_[1, 1, 1, 0]
        self.y_[1, 1] = 1 * self.wt_[1, 1, 0, 0] + \
            2 * self.wt_[1, 1, 1, 0] + \
            1 * self.wt_[1, 1, 1, 1]

        # x_
        self.x_[0] = self.y_[0, 0] + self.y_[0, 1]
        self.x_[1] = self.y_[1, 0] + self.y_[1, 1]

        if np.absolute(self.eta_sum - eta_sum) > 1e-12:  # e-10 is needed
            self.__run()
