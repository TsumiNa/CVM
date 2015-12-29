#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def __eta_dt(self, i, j, k, l, m, n, o):
    """
    η_ijklmnt = exp[-β*e_ijkl +
                    (β/168)(mu_i + mu_j + mu_k + mu_l + mu_m + mu_n + mu_o)]
                * M61^(1/2)
                * M51^(-1/8)
                * M52^(1/2)
                * M41^(-1/12)
                * M42^(-1/4)
                * M21^(-3/44)
                * M22^(-1/32)
                * M23^(1/8)
                * X^(-1/21)
    -------------------------------------------------------------------------
    M61 = m51_ijkmno * m51_ijlmno
    M51 = m51_ijmno
    M52 = m52_ijklm * m52_ijklo
    M41 = m41_ijkl
    M42 = m42_ijkm * m42_ijlo
    M21 = m21_ij * m21_ik * m21_il *
          m21_jk * m21_jl * m21_jm * m21_jn * m21_jo *
          m21_ol * m21_lk * m21_km
    M22 = m22_im * m22_io * m22_nm * m22_no
    M23 = m23_kn * m23_ko * m23_lm * m23_ln
    X = x_i * x_j * x_k * x_l * x_m * x_n * x_o
    """
    # exp
    exp = np.exp(-self.beta * self.enTS[i, j, k, l, m, n, o] +
                 (self.beta / 168) *
                 (self.mu[i] + self.mu[j] + self.mu[k] + self.mu[l] +
                  self.mu[m] + self.mu[n] + self.mu[o]))

    # M61
    M61 = self.m61_[i, j, k, m, n, o] * self.m61_[o, j, l, i, m, n]

    # M51
    M51 = self.m51_[i, j, m, n, o]

    # M52
    M52 = self.m52_[i, j, l, k, m] * self.m52_[i, j, k, l, o]

    # M41
    M41 = self.m41_[i, j, k, l]

    # M42
    M42 = self.m42_[i, j, k, m] * self.m42_[i, j, l, o]

    # M21
    M21 = self.m21_[i, j] * self.m21_[i, k] * self.m21_[i, l] * \
        self.m21_[j, k] * self.m21_[j, l] * self.m21_[j, m] * \
        self.m21_[j, n] * self.m21_[j, o] * self.m21_[k, l] * \
        self.m21_[k, m] * self.m21_[l, o]

    # M22
    M22 = self.m22_[i, m] * self.m22_[i, o] * self.m22_[m, n] * self.m22_[n, o]

    # M23
    M23 = self.m23_[k, n] * self.m23_[k, o] * self.m23_[l, m] * self.m23_[l, n]

    # X
    X = self.x_[i] * self.x_[j] * self.x_[k] * self.x_[l] *\
        self.x_[m] * self.x_[n] * self.x_[o]

    return (exp *
            np.power(M61, 1 / 2) *
            np.power(M51, -1 / 8) *
            np.power(M52, 1 / 2) *
            np.power(M41, -1 / 12) *
            np.power(M42, -1 / 4) *
            np.power(M21, -3 / 44) *
            np.power(M22, -1 / 32) *
            np.power(M23, 1 / 8) *
            np.power(X, -1 / 21))


def process(self):
    # counts
    self.count += 1

    # calculate eta
    eta_sum = np.float64(0)
    dt_ = np.zeros((2, 2, 2, 2, 2, 2, 2), np.float64)
    it = np.nditer(dt_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l, m, n, o = it.multi_index
        dt_[i, j, k, l, m, n, o] = __eta_dt(self, i, j, k, l, m, n, o)
        eta_sum += dt_[i, j, k, l, m, n, o]
        # print('  dt_{}: {:0<.8f}'.format(it.multi_index, dt_[i, j, k, l, m, n, o]))
        it.iternext()

    # normalization
    self.checker = np.float64(0)
    self.x_ = np.zeros((2), np.float64)

    # pair
    self.m21_ = np.zeros((2, 2), np.float64)
    self.m22_ = np.zeros((2, 2), np.float64)
    self.m23_ = np.zeros((2, 2), np.float64)
    m23_ = np.zeros((2, 2), np.float64)

    # 4-body
    self.m41_ = np.zeros((2, 2, 2, 2), np.float64)
    self.m42_ = np.zeros((2, 2, 2, 2), np.float64)

    # 5-body
    self.m51_ = np.zeros((2, 2, 2, 2, 2), np.float64)
    self.m52_ = np.zeros((2, 2, 2, 2, 2), np.float64)

    # 6-body
    self.m61_ = np.zeros((2, 2, 2, 2, 2, 2), np.float64)
    m61_ = np.zeros((2, 2, 2, 2, 2, 2), np.float64)

    it = np.nditer(dt_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l, m, n, o = it.multi_index
        # print('self.zt_{} is: {}'.format(it.multi_index, self.zt_[i, j, k]))
        dt_[i, j, k, l, m, n, o] /= eta_sum
        delta = (dt_[i, j, k, l, m, n, o] - self.dt_[i, j, k, l, m, n, o])
        self.checker += np.absolute(delta)

        # dt_
        self.dt_[i, j, k, l, m, n, o] = dt_[i, j, k, l, m, n, o]
            # (2 * dt_[i, j, k, l, m, n, o] + self.dt_[i, j, k, l, m, n, o]) / 3

        # m61_
        self.m61_[i, j, k, m, n, o] += self.dt_[i, j, k, l, m, n, o]
        m61_[o, j, l, i, m, n] += self.dt_[i, j, k, l, m, n, o]

        # m51_
        self.m51_[i, j, m, n, o] += self.dt_[i, j, k, l, m, n, o]

        # m52_
        self.m52_[i, j, l, k, m] += self.dt_[i, j, k, l, m, n, o]

        # m41_
        self.m41_[i, j, k, l] += self.dt_[i, j, k, l, m, n, o]

        # m42_
        self.m42_[i, j, k, m] += self.dt_[i, j, k, l, m, n, o]

        # m21_
        self.m21_[i, j] += self.dt_[i, j, k, l, m, n, o]

        # m22_
        self.m22_[i, m] += self.dt_[i, j, k, l, m, n, o]

        # m23_
        self.m23_[k, n] += self.dt_[i, j, k, l, m, n, o]
        m23_[l, m] += self.dt_[i, j, k, l, m, n, o]

        # x_
        self.x_[i] += self.dt_[i, j, k, l, m, n, o]
        it.iternext()

    # print('  chker: {:0<8.4g},   condition: {:0<8.2g},   x1: {:0<8.4g},  eta_sum:  {:0<8.4g}'
    #       .format(self.checker, self.condition, self.x_[1], eta_sum))

    it = np.nditer(self.m23_, flags=['multi_index'])
    while not it.finished:
        i, j = it.multi_index
        print('  self.m23_{}:  {:0<8.8f}'
              .format(it.multi_index, self.m23_[i, j]))
        print('  m23_{}:       {:0<8.8f} \n'
              .format(it.multi_index, m23_[i, j]))
        it.iternext()
