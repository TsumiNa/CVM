#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def __eta_dt(self, i, j, k, l, m, n):
    """
    η_ijklmn = exp[-β*e_ijklmn +
                    (β/36)(mu_i + mu_j + mu_k + mu_l + mu_m + mu_n)]
                * M41^(5/6)
                * M31^(1/2)
                * M21^(-4/11)
                * M22^(-1/4)
                * X^(13/36)
    -------------------------------------------------------------------------
    M41 = m41_ijkl * m41_klmn
    M31 = m31_imk * m31_iml * m31_jnk * m31_jnl
    M21 = m21_ij * m21_ik * m21_il * m21_jk * m21_jl *
          m21_kl * m21_km * m21_kn *
          m21_lm * m21_ln * m21_mn
    M22 = m22_jn * m22_im
    X = x_i * x_j * x_k * x_l * x_m * x_n
    """
    # exp
    exp = np.exp(-self.beta * self.enDT[i, j, k, l, m, n] +
                 (self.beta / 36) *
                 (self.mu[i] + self.mu[j] + self.mu[k] +
                  self.mu[l] + self.mu[m] + self.mu[n]))

    # M41
    M41 = self.m41_[i, j, k, l] * self.m41_[m, n, k, l]

    # M31
    M31 = self.m31_[i, m, k] * self.m31_[i, m, l] * \
        self.m31_[j, n, k] * self.m31_[j, n, l]

    # M21
    M21 = self.m21_[i, j] * self.m21_[i, k] * self.m21_[i, l] * \
        self.m21_[j, k] * self.m21_[j, l] * self.m21_[k, l] * \
        self.m21_[k, m] * self.m21_[k, n] * self.m21_[l, m] * \
        self.m21_[l, n] * self.m21_[m, n]

    # M22
    M22 = self.m22_[j, n] * self.m22_[i, m]

    # X
    X = self.x_[i] * self.x_[j] * self.x_[k] * \
        self.x_[l] * self.x_[m] * self.x_[n]

    return (exp *
            np.power(M41, 5 / 6) *
            np.power(M31, 1 / 2) *
            np.power(M21, -4 / 11) *
            np.power(M22, -1 / 4) *
            np.power(X, 13 / 36))


def process(self):
    # counts
    self.count += 1

    # calculate eta
    eta_sum = np.float64(0)
    dt_ = np.zeros((2, 2, 2, 2, 2, 2), np.float64)
    it = np.nditer(dt_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l, m, n = it.multi_index
        dt_[i, j, k, l, m, n] = __eta_dt(self, i, j, k, l, m, n)
        eta_sum += dt_[i, j, k, l, m, n]
        it.iternext()

    ############################
    # normalization
    ############################
    self.checker = np.float64(0)

    # 4-body
    self.m41_ = np.zeros((2, 2, 2, 2), np.float64)

    # 3-body
    self.m31_ = np.zeros((2, 2, 2), np.float64)

    # pair
    self.m21_ = np.zeros((2, 2), np.float64)
    self.m22_ = np.zeros((2, 2), np.float64)
    m22_ = np.zeros((2, 2), np.float64)

    # point
    self.x_ = np.zeros((2), np.float64)

    it = np.nditer(dt_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l, m, n = it.multi_index
        # print('self.zt_{} is: {}'.format(it.multi_index, self.zt_[i, j, k]))
        dt_[i, j, k, l, m, n] /= eta_sum
        self.checker += np.absolute(dt_[i, j, k, l, m, n] -
                                    self.dt_[i, j, k, l, m, n])

        # dt_
        self.dt_[i, j, k, l, m, n] = dt_[i, j, k, l, m, n]

        # m41_
        self.m41_[i, j, k, l] += self.dt_[i, j, k, l, m, n]

        # m31_
        self.m31_[i, m, k] += self.dt_[i, j, k, l, m, n]

        # m21_
        self.m21_[i, j] += self.dt_[i, j, k, l, m, n]

        # m22_
        self.m22_[j, n] += self.dt_[i, j, k, l, m, n]
        m22_[i, m] += self.dt_[i, j, k, l, m, n]

        # x_
        self.x_[i] += self.dt_[i, j, k, l, m, n]
        it.iternext()

    # print('  chker: {:0<8.6f},   condition: {:0<8.2g},   x1: {:0<8.4f},  eta_sum:  {:0<8.4f}'
    #       .format(self.checker, self.condition, self.x_[1], eta_sum))

    # it = np.nditer(self.m22_, flags=['multi_index'])
    # while not it.finished:
    #     i, j = it.multi_index
    #     print('  self.m22_{}:  {:0<8.8f}'
    #           .format(it.multi_index, self.m22_[i, j]))
    #     print('  m22_{}:       {:0<8.8f} \n'
    #           .format(it.multi_index, m22_[i, j]))
    #     it.iternext()
