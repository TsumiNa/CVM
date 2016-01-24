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
        #     p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14 = \
        #         it.multi_index
        #     self.enFP[p1, p2, p3, p4, p5, p6, p7, p8,
        #               p9, p10, p11, p12, p13, p14] = \
        #         (1 / 2) * (e1[p1, p2] + e1[p1, p3] + e1[p1, p4] + e1[p2, p3] +
        #                    e1[p2, p4] + e1[p3, p4] + e1[p3, p5] + e1[p3, p6] +
        #                    e1[p4, p5] + e1[p4, p6] + e1[p5, p6]) + \
        #         (1 / 4) * (de22[p1, p5] + de22[p2, p6]) + \
        #         (1 / 1) * (de23[p1, p6] + de23[p2, p5]) + \
        #         (1 / 3) * (de31[p1, p2, p3] + de31[p1, p3, p4] +
        #                    de31[p3, p4, p1] + de31[p3, p4, p2] +
        #                    de31[p5, p6, p3] + de31[p5, p6, p4] +
        #                    de31[p3, p4, p6] + de31[p3, p4, p5]) + \
        #         (1 / 3) * de41[p1, p2, p3, p4]
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
            p1, p2, p3, p4, p5, p6 = it.multi_index

            # m61_
            self.m61_[p1, p2, p3, p4, p5, p6] = \
                self.x_[p1] * self.x_[p2] * self.x_[p3] * \
                self.x_[p4] * self.x_[p5] * self.x_[p6]

            # m51_
            self.m51_[p1, p2, p3, p4, p5] = \
                self.x_[p1] * self.x_[p2] * self.x_[p3] * \
                self.x_[p4] * self.x_[p5]

            # m41_
            self.m41_[p1, p2, p3, p4] = \
                self.x_[p1] * self.x_[p2] * \
                self.x_[p4] * self.x_[p5]

            # m31_
            self.m31_[p1, p2, p3] = \
                self.x_[p1] * self.x_[p2] * self.x_[p3]

            # m21_
            self.m21_[p1, p2] = self.x_[p1] * self.x_[p2]

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
