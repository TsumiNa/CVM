#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def __eta_qt(self, i, j, k, l, m, n, o, p, q):
    """
    η_ijklmnopq = exp[-β*e_ijklmn +
                    (β/54)(mu_i + mu_j + mu_k + mu_l + mu_m + mu_n + mu_o + mu_p + mu_q)]
                * m61^(3/4)
                * m51^(1/2)
                * m41^(-7/12)
                * m42^(1/2)
                * m31^(-2/7)
                * m21^(1/5)
                * m22^(1/12)
                * x^(-4/27)
    -------------------------------------------------------------------------
    m61 = ijkqno * kjlnop * ljmopq * mjipqn
    m51 = ijklm
    m41 = ijqn * kjno * ljop * mjpq
    m42 = nopq
    m31 = [ ijk * kjl * ljm * mji ] *
          [ ink * kol * lpm * mqi ] *
          [ qno * qjo * qpo ] *
          [ nop * njp * nqp ]
    m21 = ji * jk * jl * jm * jn * jo * jp * jq * iq * in *
          kn * ko * lo * lp * mp * mq * nq * no * pq * po
    m22 = [ ik * kl * lm * mi ] *
          [ np * oq ]
    x = i * j * k * l * m * n * o * p * q
    """
    # exp
    exp = np.exp(-self.beta * self.enQT[i, j, k, l, m, n, o, p, q] +
                 (self.beta / 54) *
                 (self.mu[i] + self.mu[j] + self.mu[k] +
                  self.mu[l] + self.mu[m] + self.mu[n] +
                  self.mu[o] + self.mu[p] + self.mu[q]))

    # m61
    m61 = self.m61_[i, j, k, l, m, n] * self.m61_[m, j, n, k, o, p] * \
        self.m61_[o, j, p, n, q, l] * self.m61_[q, j, l, p, i, k]

    # m51
    m51 = self.m51_[i, j, m, o, q]

    # m41
    m41 = self.m41_[i, j, k, l] * self.m41_[m, j, n, k] * \
        self.m41_[o, j, p, n] * self.m41_[q, j, l, p]

    # m42
    m42 = self.m42_[k, n, p, l]

    # m31
    m31 = (self.m31_[i, j, m] * self.m31_[m, j, o] *
           self.m31_[o, j, q] * self.m31_[q, j, i]) * \
        (self.m31_[i, k, m] * self.m31_[m, n, o] *
         self.m31_[o, p, q] * self.m31_[q, l, i]) * \
        (self.m31_[l, j, n] * self.m31_[k, j, p]) * \
        (self.m31_[k, n, p] * self.m31_[n, p, l] *
         self.m31_[p, l, k] * self.m31_[l, k, n])

    # m21
    m21 = (self.m21_[i, j] * self.m21_[m, j] *
           self.m21_[o, j] * self.m21_[q, j]) * \
        (self.m21_[i, k] * self.m21_[m, n] *
         self.m21_[o, p] * self.m21_[q, l]) * \
        (self.m21_[i, l] * self.m21_[m, k] *
         self.m21_[o, n] * self.m21_[q, p]) * \
        (self.m21_[k, j] * self.m21_[n, j] *
         self.m21_[p, j] * self.m21_[l, j]) * \
        (self.m21_[l, k] * self.m21_[k, n] *
         self.m21_[n, p] * self.m21_[p, l])

    # m22
    m22 = (self.m22_[i, m] * self.m22_[m, o] *
           self.m22_[o, q] * self.m22_[q, i]) * \
        (self.m22_[l, n] * self.m22_[k, p])

    # x
    x = self.x_[i] * self.x_[m] * self.x_[o] * self.x_[q] * \
        self.x_[j] * \
        self.x_[k] * self.x_[n] * self.x_[p] * self.x_[l]

    return (exp *
            np.power(m61, 3 / 4) *
            np.power(m51, 1 / 2) *
            np.power(m41, -7 / 12) *
            np.power(m42, 1 / 2) *
            np.power(m31, -2 / 7) *
            np.power(m21, 1 / 5) *
            np.power(m22, 1 / 12) *
            np.power(x, -4 / 27)
            )


def process(self):
    # counts
    self.count += 1

    # calculate eta
    eta_sum = np.float64(0)
    qt_ = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2), np.float64)
    it = np.nditer(qt_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l, m, n, o, p, q = it.multi_index
        qt_[i, j, k, l, m, n, o, p, q] = \
            __eta_qt(self, i, j, k, l, m, n, o, p, q)
        eta_sum += qt_[i, j, k, l, m, n, o, p, q]
        it.iternext()

    ############################
    # normalization
    ############################
    self.checker = np.float64(0)

    # 6-body
    self.m61_ = np.zeros((2, 2, 2, 2, 2, 2), np.float64)

    # 5-body
    self.m51_ = np.zeros((2, 2, 2, 2, 2), np.float64)

    # 4-body
    self.m41_ = np.zeros((2, 2, 2, 2), np.float64)

    # 4-body
    self.m42_ = np.zeros((2, 2, 2, 2), np.float64)

    # 3-body
    self.m31_ = np.zeros((2, 2, 2), np.float64)

    # pair-1st
    self.m21_ = np.zeros((2, 2), np.float64)

    # pair-2nd
    self.m22_ = np.zeros((2, 2), np.float64)

    # point
    self.x_ = np.zeros((2), np.float64)

    it = np.nditer(qt_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l, m, n, o, p, q = it.multi_index
        # print('self.zt_{} is: {}'.format(it.multi_index, self.zt_[i, j, k]))
        qt_[i, j, k, l, m, n, o, p, q] /= eta_sum
        delta = qt_[i, j, k, l, m, n, o, p, q] - \
            self.qt_[i, j, k, l, m, n, o, p, q]
        self.checker += np.absolute(delta)

        # qt_
        self.qt_[i, j, k, l, m, n, o, p, q] = \
            qt_[i, j, k, l, m, n, o, p, q]

        # m61_
        self.m61_[i, j, k, l, m, n] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m51_
        self.m51_[i, j, m, o, q] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m41_
        self.m41_[i, j, k, l] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m42_
        self.m42_[k, n, p, l] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m31_
        self.m31_[i, k, m] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m21_
        self.m21_[i, j] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m22_
        self.m22_[k, p] += self.qt_[i, j, k, l, m, n, o, p, q]

        # x_
        self.x_[i] += self.qt_[i, j, k, l, m, n, o, p, q]
        it.iternext()

    print('  chker: {:0<8.6f},   condition: {:0<8.2g},   x1: {:0<8.4f},  eta_sum:  {:0<8.4f}'
          .format(self.checker, self.condition, self.x_[1], eta_sum))

    it = np.nditer(self.m21_, flags=['multi_index'])
    while not it.finished:
        i, j = it.multi_index
        print('  self.m21_{}:  {:0<8.8f}'
              .format(it.multi_index, self.m21_[i, j]))
        it.iternext()
