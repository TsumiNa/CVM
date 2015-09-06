#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from copy import deepcopy
from numpy.matlib import zeros


class process(object):

    """docstring for process"""

    __slots__ = ('data',
                 'beta',  # beat = 1/kt
                 'lam',  # λ Lagrange multiplier
                 'x_i',  # concentration
                 'eta_ij',  # median value
                 'e_ij',  # interaction energy
                 'mu_ij',  # opposite chemical potential
                 'omega',  # coordination number
                 'omega_sum',  # sum of omega
                 'count'  # count
                 )

    def __init__(self, data):
        super(process, self).__init__()
        self.count = 0

        # init
        self.x_i = data.x_i
        self.omega = data.omega  # np.array
        self.omega_sum = self.omega.sum()
        self.beta = np.float64(pow(data.k * data.temp, -1))
        self.lam = np.zeros(self.omega.size, dtype=np.float64)
        self.mu_ij = np.matrix([[2 * data.mu_ij, 0],
                                [0, -2 * data.mu_ij]])

        # init e_ij
        self.e_ij = np.zeros(self.omega.size, dtype='(2,2)float64')
        for t in range(self.omega.size):
            self.e_ij[t] = -0.5 * np.matrix([[0., data.e_ij[t]],
                                             [data.e_ij[t], 0.]])

        # run
        self.__run()
        print(self.count)
        data.output['x_i'] = np.array(self.x_i).reshape(2).tolist()

    def __eta_ij(self, t):
        """
        η^t_ij = (x_i*x_j)^((2Σω^t-1)/2ω^t) * exp( -βe^t_ij + (β/2ω^t)(mu_i + mu_j) )
        """
        entro = np.power((self.x_i * self.x_i.T),
                         ((2 * self.omega_sum - 1) / (2 * self.omega[t])))
        energy = np.exp(-self.beta * np.matrix(self.e_ij[t]) +
                        (self.beta / (2 * self.omega[t])) * self.mu_ij)
        return np.multiply(entro, energy)

    def __y_ij(self):
        """
        y_ij = η_ij * exp(β*λ/ω)
        """
        eta_ij = np.zeros(self.omega.size, dtype='(2,2)float64')
        y_ij = np.zeros(self.omega.size, dtype='(2,2)float64')
        eta_sum = np.zeros(self.omega.size)
        for t in range(self.omega.size):
            eta_ij[t] = self.__eta_ij(t)
            eta_sum[t] = np.sum(eta_ij[t])
            self.lam[t] = -np.log(eta_sum[t]) * self.omega[t] / self.beta
            y_ij[t] = eta_ij[t] * np.power(eta_sum[t], -1)
        return y_ij

    def __run(self):
        """
        TODO: need doc
        """
        lam = deepcopy(self.lam)
        self.count += 1
        y_ij = self.__y_ij()
        print(y_ij)
        print('lambda is: {}'.format(self.lam))
        print('old lambda is: {}'.format(lam))
        self.x_i.fill(0)
        for t in range(self.omega.size):
            self.x_i += np.matrix(y_ij[t]).sum(1)
        print(self.x_i)
        print('\n')
        if np.absolute(self.lam.sum() - lam.sum()) > 1e-6:
            self.__run()
