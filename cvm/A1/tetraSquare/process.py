#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def __eta_ts(self, i, j, k, l, m, n, o):
    """
    η_ijklmnt = exp[-β*e_ijkl +
                    (β/168)(mu_i + mu_j + mu_k + mu_l + mu_m + mu_n + mu_o)]
                * M1^(-1/21)
                * M21^(-1/44)
                * M22^(-5/32)
                * M4^(-1/12)
                * M51^(7/8)
                * M52^(1/2)
    M1 = m1_i * m1_j * m1_k * m1_l * m1_m * m1_n * m1_o
    M21 = m21_ij * m21_ik * m21_il *
          m21_jk * m21_jl * m21_jm * m21_jn * m21_jo *
          m21_ol * m21_lk * m21_km
    M22 = m22_im * m22_io * m22_nm * m22_no
    M4 = m4_ijkl
    M51 = m51_ijmno
    M52 = m52_ijklm * m52_ijklo
    """
    # exp
    exp = np.exp(-self.beta * self.enTS[i, j, k, l, m, n, o] +
                 (self.beta / 168) *
                 (self.mu[i] + self.mu[j] + self.mu[k] + self.mu[l] +
                  self.mu[m] + self.mu[n] + self.mu[o]))

    # M1
    M1 = self.m1_[i] * self.m1_[j] * self.m1_[k] * self.m1_[l] *\
        self.m1_[m] * self.m1_[n] * self.m1_[o]

    # M21
    M21 = self.m21_[i, j] * self.m21_[i, k] * self.m21_[i, l] * \
        self.m21_[j, k] * self.m21_[j, l] * self.m21_[j, m] * \
        self.m21_[j, n] * self.m21_[j, o] * self.m21_[o, l] * \
        self.m21_[l, k] * self.m21_[k, m]

    # M22
    M22 = self.m22_[i, m] * self.m22_[i, o] * self.m22_[n, m] * self.m22_[n, o]

    # M4
    M4 = self.m4_[i, j, k, l]

    # M51
    M51 = self.m51_[i, j, m, n, o]

    # M52
    M52 = self.m52_[i, j, l, k, m] * self.m52_[i, j, k, l, o]

    return exp * np.power(M1, -1 / 21) * \
        np.power(M21, -1 / 44) * np.power(M22, -5 / 32) * \
        np.power(M4, -1 / 12) * \
        np.power(M51, 7 / 8) * np.power(M52, 1 / 2)


def process(self):
    # counts
    self.count += 1

    # calculate eta
    eta_sum = np.float64(0)
    ts_ = np.zeros((2, 2, 2, 2, 2, 2, 2), np.float64)
    it = np.nditer(ts_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l, m, n, o = it.multi_index
        ts_[i, j, k, l, m, n, o] = __eta_ts(self, i, j, k, l, m, n, o)
        eta_sum += ts_[i, j, k, l, m, n, o]
        it.iternext()

    # normalization
    self.checker = np.float64(0)
    self.m1_ = np.zeros((2), np.float64)
    self.m21_ = np.zeros((2, 2), np.float64)
    self.m22_ = np.zeros((2, 2), np.float64)
    self.m4_ = np.zeros((2, 2, 2, 2), np.float64)
    self.m51_ = np.zeros((2, 2, 2, 2, 2), np.float64)
    self.m52_ = np.zeros((2, 2, 2, 2, 2), np.float64)
    it = np.nditer(ts_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l, m, n, o = it.multi_index
        # print('self.zt_{} is: {}'.format(it.multi_index, self.zt_[i, j, k]))
        ts_[i, j, k, l, m, n, o] /= eta_sum
        delta = ts_[i, j, k, l, m, n, o] - self.ts_[i, j, k, l, m, n, o]
        self.checker += np.absolute(delta * 0.3)

        # ts_
        self.ts_[i, j, k, l, m, n, o] += delta * 0.3

        # m51_
        self.m51_[i, j, m, n, o] += self.ts_[i, j, k, l, m, n, o]

        # m52_
        self.m52_[i, j, m, k, l] += self.ts_[i, j, k, l, m, n, o]

        # m4_
        self.m4_[i, j, k, l] += self.ts_[i, j, k, l, m, n, o]

        # m21_
        self.m21_[i, j] += self.ts_[i, j, k, l, m, n, o]

        # m22_
        self.m22_[i, m] += self.ts_[i, j, k, l, m, n, o]

        # m1_
        self.m1_[i] += self.ts_[i, j, k, l, m, n, o]
        it.iternext()

    print('  chker: {:0<8.4g},   condition: {:0<8.2g},   x1: {:0<8.4g},  eta_sum:  {:0<8.4g}'
          .format(self.checker, self.condition, self.m1_[1], eta_sum))
