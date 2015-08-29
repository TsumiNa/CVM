#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
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
                 'count'  # count
                 )

    def __init__(self, data):
        super(process, self).__init__()
        self.count = 0

        # init
        self.x_i = data.x_i
        self.omega = data.omega[0, 0]  # TODO: now only the first neighbour
        self.beta = np.float64(pow(data.k * data.temp, -1))
        self.lam = np.float64(0.0)
        self.eta_ij = zeros((data.x_i.size, data.x_i.size))
        self.e_ij = np.matrix([[0., data.e_ij[0, 0]],
                               [data.e_ij[0, 0], 0.]])
        self.mu_ij = np.matrix([[2 * data.mu_ij, 0],
                                [0, -2 * data.mu_ij]])
        self.e_ij = -0.5 * self.e_ij

        # run
        self.__run()
        print(self.count)
        data.output['x_i'] = np.array(self.x_i).reshape(2).tolist()

    def __eta_ij(self):
        """
        η_ij = (x_i*x_j)^((2ω-1)/2ω) * exp( -βe_ij + (β/2ω)(mu_i + mu_j) )
        """
        entro = np.power((self.x_i * self.x_i.T),
                         ((2 * self.omega - 1) / (2 * self.omega)))
        energy = np.exp(-self.beta * self.e_ij +
                        (self.beta / (2 * self.omega)) * self.mu_ij)
        return np.multiply(entro, energy)

    def __y_ij(self):
        """
        y_ij = η_ij * exp(β*λ/ω)
        """
        eta_ij = self.__eta_ij()
        eta_sum = np.sum(eta_ij)
        self.lam = -np.log(eta_sum) * self.omega / self.beta
        return eta_ij * np.power(eta_sum, -1)

    def __run(self):
        """
        TODO: need doc
        """
        lam = self.lam
        y_ij = self.__y_ij()
        self.count += 1
        print('lambda is: {}'.format(self.lam))
        self.x_i = y_ij.sum(1)
        print(y_ij)
        print(self.x_i)
        print('\n')
        if np.absolute(self.lam - lam) > 1e-6:
            self.__run()
