#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from cvm.utilities import CVM


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

        en1 = np.zeros((2, 2), np.float64)
        en1[0, 1] = en1[0, 1] = 0.5 * (en1[0, 0] + en1[1, 1] - self.int_pair)
        self.enT[0, 0, 0, 0] = 0.0
        self.enT[1, 0, 0, 0] = 1.5 * (en1[0, 0] + en1[0, 1])
        self.enT[1, 1, 0, 0] = \
            0.5 * (en1[0, 0] + 4 * en1[0, 1] + en1[1, 1])
        self.enT[1, 1, 1, 0] = 1.5 * (en1[0, 1] + en1[1, 1]) + self.int_trip
        self.enT[1, 1, 1, 1] = \
            3.0 * en1[1, 1] + 4 * self.int_trip + self.int_tetra

        self.mu[0] = self.enT[0, 0, 0, 0] - self.enT[1, 1, 1, 1]
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
