#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def __eta_ijkl(self, i, j, k, l, m, n):
    """
    η_ijkl = exp[-β*e_ijkl + (β/6)(mu_i + mu_j + mu_k + mu_l + mu_m + mu_n)]
                * X^(1/6)
                * Y^(-1/2)
                * Z^(1/1)
    X = x_i * x_j * x_k * x_l * x_m * x_n
    Y = y_ij * y_il * y_im * y_in * y_jk * y_jm *
        y_jn * y_kl * y_km * y_kn * y_lm * y_ln
    Z = z_ijm * z_ijn * z_jkm * z_jkn * z_klm * z_kln * z_ilm * z_iln
    """
    # exp
    exp = np.exp(-self.beta * self.enO[i, j, k, l] +
                 (self.beta / 8) *
                 (self.mu[i] + self.mu[j] +
                  self.mu[k] + self.mu[l]))

    # X
    X = self.x_[i] * self.x_[j] * self.x_[k] *\
        self.x_[l] * self.x_[m] * self.x_[n]

    # Y
    Y = self.y_[i, j] * self.y_[i, l] * self.y_[i, m] * self.y_[i, n] *\
        self.y_[j, k] * self.y_[j, m] * self.y_[j, n] * self.y_[k, l] *\
        self.y_[k, m] * self.y_[k, n] * self.y_[l, m] * self.y_[l, n]

    # Z
    Z = self.z_[i, j, m] * self.z_[i, j, n] * self.z_[j, k, m] *\
        self.z_[j, k, n] * self.z_[k, l, m] * self.z_[k, l, n] *\
        self.z_[i, l, m] * self.z_[i, l, n]

    return exp * np.power(X, 1 / 8) * np.power(Y, -1 / 2) * Z


def __z_ijkl(self):
    """
    z_ijkl = η_ijkl * exp(β*λ)
    """
    eta_000000 = self.__eta_ijkl(0, 0, 0, 0, 0, 0)
    eta_100000 = self.__eta_ijkl(1, 0, 0, 0, 0, 0)
    eta_110000 = self.__eta_ijkl(1, 1, 0, 0, 0, 0)
    eta_111000 = self.__eta_ijkl(1, 1, 1, 0, 0, 0)
    eta_111100 = self.__eta_ijkl(1, 1, 1, 1, 0, 0)
    eta_111110 = self.__eta_ijkl(1, 1, 1, 1, 1, 0)
    eta_111111 = self.__eta_ijkl(1, 1, 1, 1, 1, 1)
    self.eta_sum = eta_111111 + eta_111000 * 4 + \
        eta_110000 * 6 + eta_100000 * 4 + eta_000000
    self.wo_[0, 0, 0, 0, 0, 0] = eta_000000 / self.eta_sum
    self.wo_[1, 0, 0, 0, 0, 0] = eta_100000 / self.eta_sum
    self.wo_[1, 1, 0, 0, 0, 0] = eta_110000 / self.eta_sum
    self.wo_[1, 1, 1, 0, 0, 0] = eta_111000 / self.eta_sum
    self.wo_[1, 1, 1, 1, 0, 0] = eta_111100 / self.eta_sum
    self.wo_[1, 1, 1, 1, 1, 0] = eta_111110 / self.eta_sum
    self.wo_[1, 1, 1, 1, 1, 1] = eta_111111 / self.eta_sum
