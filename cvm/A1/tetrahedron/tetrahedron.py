#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np

from cvm.utilities import CVM

from .process import process


class tetrahedron(CVM):
    """docstring for tetrahedron"""

    __slots__ = (
        'x_',  # concentration of point, dim is 2
        'y_',  # concentration of pair, dim is 2x2
        't_',  # median value, dim is 2x2x2x2
        'en',  # interaction energy, dim is 2x2x2x2
        'mu',  # opposite chemical potential, dim is 2
        'beta',  # β = 1/kt
        'checker',  # absolute of t_out and t_in
        'eta_sum',  # sum of η_ijkl
        'multi_calcu',  # if is a multiple calculation
    )

    def __init__(self, inp):
        super(tetrahedron, self).__init__(inp)
        ####################
        # define var
        ####################
        self.multi_calcu = True if len(inp['methods']) > 1 else False
        self.x_ = np.zeros((2), np.float64)
        self.y_ = np.zeros((2, 2), np.float64)
        self.t_ = np.zeros((2, 2, 2, 2), np.float64)
        self.en = np.zeros((2, 2, 2, 2), np.float64)
        self.beta = np.float64(0.0)
        self.mu = np.zeros((2), np.float64)
        self.eta_sum = np.float64(0.0)

    def __init__en(self, r_0, T, sample):
        ###############################################
        # configuration
        ###############################################
        # use transfer
        # transfer to 2nd
        if hasattr(sample, 'transfer'):
            sample.effctive_en(
                sample.transfer[0],
                sample.transfer[1],
                sample.transfer[2], )
        else:
            sample.effctive_en(1, 8)

        # pure energy of 2body 1st
        e1 = np.zeros((2, 2), np.float64)
        e1[0, 1] = e1[1, 0] = 0.5 * (
            e1[0, 0] + e1[1, 1] - sample.int_pair_1(r_0, T))

        # 3body-1st interaction energy
        de31 = np.zeros((2, 2, 2), np.float64)
        de31[1, 1, 1] = sample.int_trip(r_0, T)

        # 4body-1st interaction energy
        de41 = np.zeros((2, 2, 2, 2), np.float64)
        de41[1, 1, 1, 1] = sample.int_tetra(r_0, T)

        # energy ε
        it = np.nditer(self.en, flags=['multi_index'])
        while not it.finished:
            i, j, k, l = it.multi_index
            self.en[i, j, k, l] = \
                0.5 * (e1[i, j] + e1[i, k] + e1[i, l] +
                       e1[j, k] + e1[j, l] + e1[k, l]) + \
                de31[i, j, k] + de31[i, k, l] + \
                de31[i, j, l] + de31[j, k, l] + \
                de41[i, j, k, l]
            # print('en{} is: {}'.format(it.multi_index, self.en[i, j, k, l]))
            it.iternext()

        # chemical potential
        self.mu[0] = (self.en[0, 0, 0, 0] - self.en[1, 1, 1, 1])
        self.mu[1] = -self.mu[0]

    def __reset__probability(self):
        """
        initialize
        """
        self.count = 0
        self.checker = np.float64(1.0)

        it = np.nditer(self.y_, flags=['multi_index'])
        while not it.finished:
            i, j = it.multi_index
            self.y_[i, j] = self.x_[i] * self.x_[j]
            it.iternext()

    # implement the inherited abstract method run()
    def run(self):

        # temperature iteration
        for sample in self.series:
            if self.multi_calcu:
                sample.res['label'] = sample.res['label'] + '(T)'
            self.x_[1] = sample.x_1
            self.x_[0] = 1 - sample.x_1

            # delta mu iteration
            for temperture in np.nditer(sample.temp):
                r_0 = temperture[0]
                temp = temperture[1]
                self.beta = np.float64(pow(self.bzc * temp, -1))

                # calculate
                self.__init__en(r_0, temp, sample)
                self.__reset__probability()
                print(' mu = {:06.4f}:'.format(self.mu[0].item(0)))
                print(' 1st_int = {:06.4f}:'.format(sample.int_pair_1(r_0, temp)))
                while self.checker > sample.condition:
                    process(self)

                # push result into res
                sample.res['c'].append(self.x_[1].item(0))
                print('    T = {:06.3f}K,  c = {:06.6f},  count = {}'.format(
                    temp.item(0), self.x_[1].item(0), self.count))

            print('\n')
            # save result to output
            self.output['results'].append(sample.res)
