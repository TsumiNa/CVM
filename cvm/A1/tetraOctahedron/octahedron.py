#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def __eta(self, i, j, k, l, m, n):
    """
    η_ijklmn = exp[-β*e_ijkl + (β/6)(mu_i + mu_j + mu_k + mu_l + mu_m + mu_n)]
                * X^(1/6)
                * Y^(-1/2)
                * Z^(1/1)
    X = x_i * x_j * x_k * x_l * x_m * x_n
    Y = y_ij * y_il * y_im * y_in * y_jk * y_jm *
        y_jn * y_kl * y_km * y_kn * y_lm * y_ln
    Z = z_ijm * z_ijn * z_jkm * z_jkn * z_klm * z_kln * z_ilm * z_iln
    """
    # exp
    exp = np.exp(-self.beta * self.enO[i, j, k, l, m, n] +
                 (self.beta / 6) *
                 (self.mu[i] + self.mu[j] + self.mu[k] +
                  self.mu[l] + self.mu[m] + self.mu[n]))

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

    return exp * np.power(X, 1 / 6) * np.power(Y, -1 / 2) * Z


def wo(self):
    """
    wo_ijklmn = η_ijklmn * exp(β*λ)
    """
    eta_sum = np.float64(0)
    it = np.nditer(self.wo_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l, m, n = it.multi_index
        self.wo_[i, j, k, l, m, n] = __eta(self, i, j, k, l, m, n)
        eta_sum += self.wo_[i, j, k, l, m, n]
        it.iternext()

    self.x_ = np.zeros((2), np.float64)
    self.y_ = np.zeros((2, 2), np.float64)
    self.z_ = np.zeros((2, 2, 2), np.float64)
    it = np.nditer(self.wo_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l, m, n = it.multi_index

        # z_ijkl = η_ijkl * exp(β*λ/2)
        self.wo_[i, j, k, l, m, n] /= eta_sum

        # z_
        self.z_[i, j, k] += self.wo_[i, j, k, l, m, n]

        # y_
        self.y_[i, j] += self.wo_[i, j, k, l, m, n]

        # x_
        self.x_[i] += self.wo_[i, j, k, l, m, n]
        it.iternext()

    print('Octa eta_sum is: {}'.format(eta_sum))
    print('Octa x_[0] is: {}'.format(self.x_[0]))
    # counts
    self.count += 1

    if np.absolute(self.eta_sum - eta_sum) > 1e-3:  # e-10 is needed
        self.eta_sum = eta_sum
        print('\n')
        wo(self)
