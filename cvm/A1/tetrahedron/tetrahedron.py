#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from cvm.utilities import CVM


class tetrahedron(CVM):

    """docstring for tetrahedron"""

    __slots__ = (
        'x_',  # concentration of point, dim is 2
        'y_',  # concentration of pair, dim is 2x2
        'z_',  # median value, dim is 2x2x2x2
        'en',  # interaction energy, dim is 2x2x2x2
        'mu',  # opposite chemical potential, dim is 2
        'beta',  # beat = 1/kt
        'eta_sum',  # sum of η_ijkl
    )

    def __init__(self, inp):
        super(tetrahedron, self).__init__(inp)
        ####################
        # define var
        ####################
        self.x_ = np.zeros((2), np.float64)
        self.y_ = np.zeros((2, 2), np.float64)
        self.z_ = np.zeros((2, 2, 2, 2), np.float64)
        self.en = np.zeros((2, 2, 2, 2), np.float64)
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

        en = np.zeros((2, 2), np.float64)
        en[0, 1] = en[0, 1] = 0.5 * (en[0, 0] + en[1, 1] - self.int_pair)
        self.en[0, 0, 0, 0] = 0.0
        self.en[1, 0, 0, 0] = 1.5 * (en[0, 0] + en[0, 1])
        self.en[1, 1, 0, 0] = 0.5 * (en[0, 0] + 4 * en[0, 1] + en[1, 1])
        self.en[1, 1, 1, 0] = 1.5 * (en[0, 1] + en[1, 1]) + self.int_trip
        self.en[1, 1, 1, 1] = \
            3.0 * en[1, 1] + 4 * self.int_trip + self.int_tetra

        self.mu[0] = self.en[0, 0, 0, 0] - self.en[1, 1, 1, 1]
        self.mu[1] = -self.mu[0]

    # implement the inherited abstract method run()
    def run(self):

        # temperature iteration
        for temp in np.nditer(self.temp):
            data = []
            self.beta = np.float64(pow(self.bzc * temp, -1))
            print('T = {:06.2f}K:'.format(temp.item(0)))

            # delta mu iteration
            for dmu in np.nditer(self.delta_mu):
                self.mu[0] += dmu
                self.__run()
                data.append(
                    {'mu': self.mu[0].item(0), 'c': self.x_[0].item(0)})
                print('    mu = {:06.4f},  c = {:06.4f}\n'.format(
                    self.mu[0].item(0), self.x_[0].item(0)))
                self.mu[0] -= dmu

            # save result to output
            self.output['Results'].append(
                {'Temp': temp.item(0), 'Data': data})

    def __eta_ijkl(self, i, j, k, l):
        """
        η_ijkl = exp[-β*e_ijkl + (β/8)(mu_i + mu_j + mu_k + mu_l)]
                    * X^(-5/8)
                    * Y^(1/2)
        X = x_ * x_j * x_k * x_l
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
        eta_2222 = self.__eta_ijkl(1, 1, 1, 1)
        eta_2111 = self.__eta_ijkl(1, 0, 0, 0)
        eta_2211 = self.__eta_ijkl(1, 1, 0, 0)
        eta_2221 = self.__eta_ijkl(1, 1, 1, 0)
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

        if np.absolute(self.eta_sum - eta_sum) > 1e-12:  # e-10 is needed
            self.__run()
