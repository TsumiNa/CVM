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

    def __init__en(self, sample):
        ###############################################
        # configuration
        ###############################################
        # use transfer
        # transfer to 2nd
        sample.effctive_en(1, 8, 3)

        # pure energy of 2body-1st
        en1 = np.zeros((2, 2), np.float64)
        en1[0, 1] = en1[1, 0] = \
            0.5 * (en1[0, 0] + en1[1, 1] - sample.int_pair[0])

        #############################################
        # tetrahedron
        #############################################
        # 3body-1st interaction energy
        de31 = np.zeros((2, 2, 2), np.float64)
        de31[1, 1, 1] = sample.int_trip[0]

        # 4body-1st interaction energy
        de41 = np.zeros((2, 2, 2, 2), np.float64)
        de41[1, 1, 1, 1] = sample.int_tetra[0]

        # energy ε
        it = np.nditer(self.enT, flags=['multi_index'])
        while not it.finished:
            i, j, k, l = it.multi_index
            self.enT[i, j, k, l] = \
                0.5 * (en1[i, j] + en1[i, k] + en1[i, l] +
                       en1[j, k] + en1[j, l] + en1[l, k]) +\
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
            0.5 * (en2[0, 0] + en2[1, 1] - sample.int_pair[1])

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
        self.mu[1] = -self.mu[0]

    def __reset__probability(self):

        self.count = 0
        self.checker = np.float64(1.0)
        self.af_ = np.zeros((2, 2, 2), np.float64)
        self.main_condition = np.float64(1e-3)
        self.sub_condition = np.float64(1e-2)

        it = np.nditer(self.z_, flags=['multi_index'])
        while not it.finished:
            i, j, k = it.multi_index
            self.z_[i, j, k] = self.x_[i] * self.x_[j] * self.x_[k]
            self.y_[i, j] = self.x_[i] * self.x_[j]
            it.iternext()

    # implement the inherited abstract method run()
    def run(self):

        # temperature iteration
        for sample in self.series:
            sample.res['temp'] = sample.temp.tolist()
            sample.res['label'] = sample.res['label'] + '(TO)'
            self.x_[1] = sample.x_1
            self.x_[0] = 1 - sample.x_1
            self.__init__en(sample)
            print(' mu = {:06.4f}:'.format(self.mu[0].item(0)))
            print(' 1st_int = {:06.4f}:'.format(sample.int_pair[0]))
            print(' 2nd_int = {:06.4f}:'.format(sample.int_pair[1]))

            # delta mu iteration
            for temp in np.nditer(sample.temp):
                self.beta = np.float64(pow(self.bzc * temp, -1))

                # calculate w
                self.__reset__probability()
                while self.checker > sample.condition:
                    while self.checker > self.main_condition:
                        process(self)
                    else:
                        self.main_condition /= 10
                        self.sub_condition /= 10

                # push result into res
                sample.res['c'].append(self.x_[1].item(0))
                print('    T = {:06.3f}K,  c = {:06.6f},  count = {}'.
                      format(temp.item(0), self.x_[1].item(0), self.count))

            print('\n')
            # save result to output
            self.output['results'].append(sample.res)
