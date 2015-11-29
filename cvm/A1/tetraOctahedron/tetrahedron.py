#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def __eta_ijkl(self, i, j, k, l):
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
    exp = np.exp(-self.beta * self.en[i, j, k, l] +
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


def __z_ijkl(self):
    """
    z_ijkl = η_ijkl * exp(β*λ/2)
    """
    eta_0000 = self.__eta_ijkl(0, 0, 0, 0)
    eta_1000 = self.__eta_ijkl(1, 0, 0, 0)
    eta_1100 = self.__eta_ijkl(1, 1, 0, 0)
    eta_1110 = self.__eta_ijkl(1, 1, 1, 0)
    eta_1111 = self.__eta_ijkl(1, 1, 1, 1)
    self.eta_sum = eta_1111 + eta_1110 * 4 + \
        eta_1100 * 6 + eta_1000 * 4 + eta_0000
    self.z_[0, 0, 0, 0] = eta_0000 / self.eta_sum
    self.z_[1, 0, 0, 0] = eta_1000 / self.eta_sum
    self.z_[1, 1, 0, 0] = eta_1100 / self.eta_sum
    self.z_[1, 1, 1, 0] = eta_1110 / self.eta_sum
    self.z_[1, 1, 1, 1] = eta_1111 / self.eta_sum
