#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def __eta(self, i, j, k, l):
    """
    η_ijkl = exp[-β*e_ijkl + (β/8)(mu_i + mu_j + mu_k + mu_l)]
                * X^(1/8)
                * Y^(-1/2)
                * Z^(1/1)
    X = x_i * x_j * x_k * x_l
    Y = y_ij * y_ik * y_il * y_jk * y_jl * y_kl
    Z = z_ijk * z_ikl * z_ijl * z_jkl
    """
    # exp
    exp = np.exp(-self.beta * self.enT[i, j, k, l] +
                 (self.beta / 8) *
                 (self.mu[i] + self.mu[j] +
                  self.mu[k] + self.mu[l]))

    # X
    X = self.x_[i] * self.x_[j] * self.x_[k] * self.x_[l]

    # Y
    Y = self.y_[i, j] * self.y_[i, k] * self.y_[i, l] * \
        self.y_[j, k] * self.y_[l, j] * self.y_[k, l]

    # Z
    Z = self.z_[i, j, k] * self.z_[i, k, l] *\
        self.z_[i, j, l] * self.z_[j, k, l]

    return exp * np.power(X, 1 / 8) * np.power(Y, -1 / 2) * Z


def wt(self):
    """
    wt_ijkl = η_ijkl * exp(β*λ/2)
    """
    eta_sum = np.float64(0)
    it = np.nditer(self.wt_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l = it.multi_index
        self.wt_[i, j, k, l] = __eta(self, i, j, k, l)
        eta_sum += self.wt_[i, j, k, l]
        it.iternext()

    self.x_ = np.zeros((2), np.float64)
    self.y_ = np.zeros((2, 2), np.float64)
    self.z_ = np.zeros((2, 2, 2), np.float64)
    it = np.nditer(self.wt_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l = it.multi_index

        # z_ijkl = η_ijkl * exp(β*λ/2)
        self.wt_[i, j, k, l] /= eta_sum

        # z_
        self.z_[i, j, k] += self.wt_[i, j, k, l]

        # y_
        self.y_[i, j] += self.wt_[i, j, k, l]

        # x_
        self.x_[i] += self.wt_[i, j, k, l]
        it.iternext()

    print('Tetra eta_sum is: {}'.format(eta_sum))
    print('Tetra x_[0] is: {}'.format(self.x_[0]))
    # counts
    self.count += 1

    if np.absolute(self.eta_sum - eta_sum) > 1e-3:  # e-10 is needed
        print('\n')
        self.eta_sum = eta_sum
        wt(self)
