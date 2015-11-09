#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


class process(object):

    """docstring for process"""

    __slots__ = ('data',
                 'beta',  # beat = 1/kt
                 'eta_sum',  # sum of η_ijkl
                 'x_',  # concentration of point, dim is 2
                 'y_',  # concentration of pair, dim is 2x2
                 'z_',  # median value, dim is 2x2x2x2
                 'en',  # interaction energy, dim is 2x2x2x2
                 'mu',  # opposite chemical potential, dim is 2
                 'omega',  # coordination number
                 'count'  # count
                 )

    def __init__(self, data):
        super(process, self).__init__()
        self.count = 0

        ####################
        # define var
        ####################
        self.mu = np.zeros((2), np.float64)
        self.en = np.zeros((2, 2, 2, 2), np.float64)
        self.x_ = np.zeros((2), np.float64)
        self.y_ = np.zeros((2, 2), np.float64)
        self.z_ = np.zeros((2, 2, 2, 2), np.float64)
        self.beta = np.float64(pow(data.bzc * data.temp, -1))
        self.eta_sum = np.float64(0.0)

        #######################
        # init
        ########################
        self.mu[0] = 0.028
        self.mu[1] = -self.mu[0]

        self.x_[0] = data.x_1
        self.x_[1] = 1 - self.x_[0]

        self.y_[0, 0] = self.x_[0]**2
        self.y_[0, 1] = self.y_[1, 0] = self.x_[0] * self.x_[1]
        self.y_[1, 1] = self.x_[1]**2

        self.en[1, 0, 0, 0] = \
            -(6 * data.int_pair + 8 * data.int_trip + 2 * data.int_tetra) / 4
        self.en[1, 1, 0, 0] = \
            -(12 * data.int_pair + 24 * data.int_trip + 6 * data.int_tetra) / 6
        self.en[1, 1, 1, 0] = \
            -(6 * data.int_pair + 16 * data.int_trip + 6 * data.int_tetra) / 4

        # run
        self.__run()

        # output
        lam = -np.log(self.eta_sum) * 2 / self.beta
        print('lambda is: {}'.format(lam))
        print('counts of self consistent: {}'.format(self.count))
        print('concentration of x is: {}'.format(self.x_))
        data.output['x_i'] = self.x_.tolist()

    def __eta_ijkl(self, i, j, k, l):
        """
        η_ijkl = exp[-β*e_ijkl + (β/8)(mu_i + mu_j + mu_k + mu_l)]
                    * X^(-5/8)
                    * Y^(1/2)
        X = x_i * x_j * x_k * x_l
        Y = y_ij * y_ik * y_il * y_jk * y_jl * y_kl
        """
        # exp
        exp = np.exp(-self.beta * self.en[i, j, k, l] +
                     (self.beta / 8) *
                     (self.mu[i] + self.mu[j] +
                      self.mu[k] + self.mu[l]))

        # X
        X = self.x_[i] * self.x_[j] * self.x_[k] * self.x_[l]

        # Y
        Y = self.y_[i, j] * self.y_[i, k] * self.y_[i, l] * \
            self.y_[j, k] * self.y_[l, j] * self.y_[k, l]

        return exp * np.power(X, -5 / 8) * np.power(Y, 1 / 2)

    def __z_ijkl(self):
        """
        z_ijkl = η_ijkl * exp(β*λ/2)
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
        self.eta_sum = eta_2222 + eta_2221 * 4 + \
            eta_2211 * 6 + eta_2111 * 4 + eta_1111
        self.z_[0, 0, 0, 0] = eta_1111 / self.eta_sum
        self.z_[1, 1, 1, 1] = eta_2222 / self.eta_sum
        self.z_[1, 0, 0, 0] = eta_2111 / self.eta_sum
        self.z_[1, 1, 0, 0] = eta_2211 / self.eta_sum
        self.z_[1, 1, 1, 0] = eta_2221 / self.eta_sum

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
        self.y_[0, 0] = 1 * self.z_[0, 0, 0, 0] + \
            2 * self.z_[1, 0, 0, 0] + \
            1 * self.z_[1, 1, 0, 0]
        self.y_[1, 0] = self.y_[0, 1] = 1 * self.z_[1, 0, 0, 0] + \
            2 * self.z_[1, 1, 0, 0] + \
            1 * self.z_[1, 1, 1, 0]
        self.y_[1, 1] = 1 * self.z_[1, 1, 0, 0] + \
            2 * self.z_[1, 1, 1, 0] + \
            1 * self.z_[1, 1, 1, 1]

        # x_
        self.x_[0] = self.y_[0, 0] + self.y_[0, 1]
        self.x_[1] = self.y_[1, 0] + self.y_[1, 1]

        print(self.y_)

        print('\n')
        if np.absolute(self.eta_sum - eta_sum) > 1e-10:  # e-10 is needed
            self.__run()
