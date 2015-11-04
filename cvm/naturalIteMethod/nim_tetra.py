#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


class process(object):

    """docstring for process"""

    __slots__ = ('data',
                 'beta',  # beat = 1/kt
                 'lam',  # λ Lagrange multiplier
                 'x_i',  # concentration
                 'y_ij',  # concentration
                 'z_ijkl',  # median value, dim is 2x2x2x2
                 'e_ijkl',  # interaction energy
                 'mu',  # opposite chemical potential
                 'omega',  # coordination number
                 'count'  # count
                 )

    def __init__(self, data):
        super(process, self).__init__()
        self.count = 0

        ####################
        # define var
        ####################
        self.x_i = np.zeros((2))
        self.y_ij = np.zeros((2, 2))
        # self.omega = data.omega[0, 0]  # TODO: now only the first neighbour
        self.beta = np.float64(pow(data.bzc * data.temp, -1))
        self.lam = np.float64(0.0)
        self.z_ijkl = np.zeros((2, 2, 2, 2))
        self.e_ijkl = np.zeros((2, 2, 2, 2))
        self.mu = np.zeros((2))

        #######################
        # init
        ########################
        self.mu[0] = -0.00
        self.mu[1] = -self.mu[0]

        self.x_i[0] = data.x_1
        self.x_i[1] = 1 - self.x_i[0]

        self.y_ij[0, 0] = self.y_ij[0, 1] = self.y_ij[1, 0] = self.x_i[0] / 2
        self.y_ij[1, 1] = self.x_i[1] - self.y_ij[1, 0]

        self.e_ijkl[1, 0, 0, 0] = \
            -(6 * data.int_pair + 8 * data.int_trip + 2 * data.int_tetra) / 4
        self.e_ijkl[1, 1, 0, 0] = \
            -(12 * data.int_pair + 24 * data.int_trip + 6 * data.int_tetra) / 6
        self.e_ijkl[1, 1, 1, 0] = \
            -(6 * data.int_pair + 16 * data.int_trip + 6 * data.int_tetra) / 4

        # run
        self.__run()
        print(self.count)
        data.output['x_i'] = self.x_i.tolist()

    def __eta_ijkl(self, i, j, k, l):
        """
        η_ijkl = exp[-β*e_ijkl + (β/8)(mu_i + mu_j + mu_k + mu_l)]
                    * X^(-5/8)
                    * Y^(1/2)
        X = x_i * x_j * x_k * x_l
        Y = y_ij * y_ik * y_il * y_jk * y_jl * y_kl
        """
        # exp
        exp = np.exp(-self.beta * self.e_ijkl[i, j, k, l] +
                     (self.beta / 8) *
                     (self.mu[i] + self.mu[j] +
                      self.mu[k] + self.mu[l]))

        # X
        X = self.x_i[i] * self.x_i[j] * self.x_i[k] * self.x_i[l]

        # Y
        Y = self.y_ij[i, j] * self.y_ij[i, k] * self.y_ij[i, l] * \
            self.y_ij[j, k] * self.y_ij[l, j] * self.y_ij[k, l]

        return exp * np.power(X, -5 / 8) * np.power(Y, 1 / 2)

    def __z_ijkl(self):
        """
        y_ij = η_ij * exp(β*λ/2)
        """
        eta_1111 = self.__eta_ijkl(0, 0, 0, 0)
        print('eta_1111 is: {}'.format(eta_1111))
        eta_2222 = self.__eta_ijkl(1, 1, 1, 1)
        print('eta_2222 is: {}'.format(eta_2222))
        eta_2111 = self.__eta_ijkl(1, 0, 0, 0)
        print('eta_2111 is: {}'.format(eta_2111))
        eta_2211 = self.__eta_ijkl(1, 1, 0, 0)
        print('eta_2211 is: {}'.format(eta_2211))
        eta_2221 = self.__eta_ijkl(1, 1, 1, 0)
        print('eta_2221 is: {}'.format(eta_2221))
        eta_sum = eta_2222 + eta_2221 * 4 + \
            eta_2211 * 6 + eta_2111 * 4 + eta_1111
        self.lam = np.log(1 / eta_sum) * 2 / self.beta
        self.z_ijkl[0, 0, 0, 0] = eta_1111 * np.power(eta_sum, -1)
        self.z_ijkl[1, 1, 1, 1] = eta_2222 * np.power(eta_sum, -1)
        self.z_ijkl[1, 0, 0, 0] = eta_2111 * np.power(eta_sum, -1)
        self.z_ijkl[1, 1, 0, 0] = eta_2211 * np.power(eta_sum, -1)
        self.z_ijkl[1, 1, 1, 0] = eta_2221 * np.power(eta_sum, -1)

    def __run(self):
        """
        TODO: need doc
        """
        lam = self.lam
        self.__z_ijkl()
        self.count += 1
        print('lambda is: {}'.format(self.lam))

        # x_i
        self.x_i[0] = 1 * self.z_ijkl[0, 0, 0, 0] + \
            3 * self.z_ijkl[1, 0, 0, 0] + \
            3 * self.z_ijkl[1, 1, 0, 0] + \
            1 * self.z_ijkl[1, 1, 1, 0]
        self.x_i[1] = 1 - self.x_i[0]

        # y_ij
        self.y_ij[0, 0] = 1 * self.z_ijkl[0, 0, 0, 0] + \
            2 * self.z_ijkl[1, 0, 0, 0] + \
            1 * self.z_ijkl[1, 1, 0, 0]
        self.y_ij[1, 0] = self.y_ij[0, 1] = 1 * self.z_ijkl[1, 0, 0, 0] + \
            2 * self.z_ijkl[1, 1, 0, 0] + \
            1 * self.z_ijkl[1, 1, 1, 0]
        self.y_ij[1, 1] = 1 * self.z_ijkl[1, 1, 0, 0] + \
            2 * self.z_ijkl[1, 1, 1, 0] + \
            1 * self.z_ijkl[1, 1, 1, 1]

        print(self.y_ij)
        print('\n')
        if np.absolute(self.lam - lam) > 1e-6:
            self.__run()
