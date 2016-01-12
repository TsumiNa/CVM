#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def __eta_ts(self, i, j, k, l, m, n, o):
    """
    η_ijklmnt = exp[-β*e_ijkl +
                    (β/168)(mu_i + mu_j + mu_k + mu_l + mu_m + mu_n + mu_o)]
                * m61^(1/2)
                * m51^(-1/8)
                * m52^(1/2)
                * m41^(-1/12)
                * m42^(-1/4)
                * m21^(-3/44)
                * m22^(-1/32)
                * m23^(1/8)
                * x^(-1/21)
    -------------------------------------------------------------------------
    m61 = m51_ijkmno * m51_ijlmno
    m51 = m51_ijmno
    m52 = m52_ijklm * m52_ijklo
    m41 = m41_ijkl
    m42 = m42_ijkm * m42_ijlo
    m21 = m21_ij * m21_ik * m21_il *
          m21_jk * m21_jl * m21_jm * m21_jn * m21_jo *
          m21_ol * m21_lk * m21_km
    m22 = m22_im * m22_io * m22_nm * m22_no
    m23 = m23_kn * m23_ko * m23_lm * m23_ln
    x = x_i * x_j * x_k * x_l * x_m * x_n * x_o
    """
    # exp
    exp = np.exp(-self.beta * self.enTS[i, j, k, l, m, n, o] +
                 (self.beta / 168) *
                 (self.mu[i] + self.mu[j] + self.mu[k] + self.mu[l] +
                  self.mu[m] + self.mu[n] + self.mu[o]))

    # m61
    m61 = self.m61_[i, j, k, m, n, o] * self.m61_[o, j, l, i, m, n]

    # m51
    m51 = self.m51_[i, j, m, n, o]

    # m52
    m52 = self.m52_[i, j, k, l, m] * self.m52_[i, l, j, k, o]

    # m41
    m41 = self.m41_[i, j, k, l]

    # m42
    m42 = self.m42_[i, j, k, m] * self.m42_[o, j, l, i]

    # m21
    m21 = self.m21_[i, j] * self.m21_[i, k] * self.m21_[i, l] * \
        self.m21_[j, k] * self.m21_[j, l] * self.m21_[j, m] * \
        self.m21_[j, n] * self.m21_[j, o] * self.m21_[k, l] * \
        self.m21_[k, m] * self.m21_[l, o]

    # m22
    m22 = self.m22_[i, m] * self.m22_[i, o] * self.m22_[m, n] * self.m22_[n, o]

    # m23
    m23 = self.m23_[k, n] * self.m23_[k, o] * self.m23_[l, m] * self.m23_[l, n]

    # x
    x = self.x_[i] * self.x_[j] * self.x_[k] * self.x_[l] *\
        self.x_[m] * self.x_[n] * self.x_[o]

    return exp * \
        np.power(m61, 1 / 2) * \
        np.power(m51, -1 / 8) * \
        np.power(m52, 1 / 2) * \
        np.power(m41, -1 / 12) * \
        np.power(m42, -1 / 4) * \
        np.power(m21, -3 / 44) * \
        np.power(m22, -1 / 32) * \
        np.power(m23, 1 / 8) * \
        np.power(x, -1 / 21)


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
        # print('  ts_{}: {:0<8.4f}'.format(it.multi_index, ts_[i, j, k, l, m, n, o]))
        it.iternext()

    # normalization
    self.checker = np.float64(0)
    self.x_ = np.zeros((2), np.float64)

    # pair
    self.m21_ = np.zeros((2, 2), np.float64)
    self.m22_ = np.zeros((2, 2), np.float64)
    self.m23_ = np.zeros((2, 2), np.float64)

    # 4-body
    self.m41_ = np.zeros((2, 2, 2, 2), np.float64)
    self.m42_ = np.zeros((2, 2, 2, 2), np.float64)

    # 5-body
    self.m51_ = np.zeros((2, 2, 2, 2, 2), np.float64)
    self.m52_ = np.zeros((2, 2, 2, 2, 2), np.float64)

    # 6-body
    self.m61_ = np.zeros((2, 2, 2, 2, 2, 2), np.float64)

    it = np.nditer(ts_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l, m, n, o = it.multi_index
        # print('self.zt_{} is: {}'.format(it.multi_index, self.zt_[i, j, k]))
        ts_[i, j, k, l, m, n, o] /= eta_sum
        delta = (ts_[i, j, k, l, m, n, o] - self.ts_[i, j, k, l, m, n, o])
        self.checker += np.absolute(delta)

        # ts_
        self.ts_[i, j, k, l, m, n, o] = (2 * ts_[i, j, k, l, m, n, o] + self.ts_[i, j, k, l, m, n, o]) / 3

        # m61_
        self.m61_[i, j, k, m, n, o] += self.ts_[i, j, k, l, m, n, o]

        # m51_
        self.m51_[i, j, m, n, o] += self.ts_[i, j, k, l, m, n, o]

        # m52_
        self.m52_[i, j, k, l, m] += self.ts_[i, j, k, l, m, n, o]

        # m41_
        self.m41_[i, j, k, l] += self.ts_[i, j, k, l, m, n, o]

        # m42_
        self.m42_[i, j, k, m] += self.ts_[i, j, k, l, m, n, o]

        # m21_
        self.m21_[i, j] += self.ts_[i, j, k, l, m, n, o]

        # m22_
        self.m22_[i, m] += self.ts_[i, j, k, l, m, n, o]

        # m23_
        self.m23_[k, n] += self.ts_[i, j, k, l, m, n, o]

        # x_
        self.x_[i] += self.ts_[i, j, k, l, m, n, o]
        it.iternext()

    print('  chker: {:0<8.4f},   condition: {:0<8.2g},   x1: {:0<8.4g},  eta_sum:  {:0<8.4g}'
          .format(self.checker, self.condition, self.x_[1], eta_sum))
