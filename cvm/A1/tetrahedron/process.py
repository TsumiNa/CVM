#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def __eta_tetra(self, i, j, k, l):
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


def process(self):
    # counts
    self.count += 1

    # calculate eta
    eta_sum = np.float64(0)
    t_ = np.zeros((2, 2, 2, 2), np.float64)
    it = np.nditer(t_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l = it.multi_index
        t_[i, j, k, l] = __eta_tetra(self, i, j, k, l)
        eta_sum += t_[i, j, k, l]
        it.iternext()

    # normalization
    self.checker = np.float64(0)
    self.x_ = np.zeros((2), np.float64)
    self.y_ = np.zeros((2, 2), np.float64)
    it = np.nditer(t_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l = it.multi_index
        t_[i, j, k, l] /= eta_sum
        self.checker += np.absolute(t_[i, j, k, l] - self.t_[i, j, k, l])

        # t_
        self.t_[i, j, k, l] = t_[i, j, k, l]

        # y_
        self.y_[i, j] += self.t_[i, j, k, l]

        # x_
        self.x_[i] += self.t_[i, j, k, l]
        it.iternext()

    # print('  chker: {:0<8.4g},   condition: {:0<8.2g},   x1: {:0<8.4g},  eta_sum:  {:0<8.4g}'
    #       .format(self.checker, self.condition, self.x_[1], eta_sum))
