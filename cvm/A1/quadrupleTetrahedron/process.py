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
    m61 = self.m61_[i, j, k, q, n, o] * self.m61_[k, j, l, n, o, p] * \
        self.m61_[l, j, m, o, p, q] * self.m61_[m, j, i, p, q, n]

    # m51
    m51 = self.m51_[i, j, k, l, m]

    # m41
    m41 = self.m41_[i, j, q, n] * self.m41_[k, j, n, o] * \
        self.m41_[l, j, o, p] * self.m41_[m, j, p, q]

    # m42
    m42 = self.m42_[n, o, p, q]

    # m31
    m31 = (self.m311_[i, j, k] * self.m311_[k, j, l] *
           self.m311_[l, j, m] * self.m311_[m, j, i]) * \
        (self.m312_[i, n, k] * self.m312_[k, o, l] *
         self.m312_[l, p, m] * self.m312_[m, q, i]) * \
        (self.m313_[q, n, o] * self.m313_[q, j, o] * self.m313_[q, p, o]) *\
        (self.m314_[n, o, p] * self.m314_[n, j, p] * self.m314_[n, q, p])

    # m21
    m21 = self.m21_[j, i] * self.m21_[j, k] * self.m21_[j, l] * \
        self.m21_[j, m] * self.m21_[j, n] * self.m21_[j, o] * \
        self.m21_[j, p] * self.m21_[j, q] * self.m21_[i, q] * \
        self.m21_[i, n] * self.m21_[k, n] * self.m21_[k, o] * \
        self.m21_[l, o] * self.m21_[l, p] * self.m21_[m, p] * \
        self.m21_[m, q] * self.m21_[n, q] * self.m21_[n, o] * \
        self.m21_[p, q] * self.m21_[p, o]

    # m22
    m22 = (self.m221_[i, k] * self.m221_[k, l] *
           self.m221_[l, m] * self.m221_[m, i]) * \
        (self.m222_[n, p] * self.m222_[o, q])

    # x
    x = self.x_[i] * self.x_[j] * self.x_[k] * \
        self.x_[l] * self.x_[m] * self.x_[n] * \
        self.x_[o] * self.x_[p] * self.x_[q]

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
    self.m311_ = np.zeros((2, 2, 2), np.float64)
    self.m312_ = np.zeros((2, 2, 2), np.float64)
    self.m313_ = np.zeros((2, 2, 2), np.float64)
    self.m314_ = np.zeros((2, 2, 2), np.float64)

    # pair-1st
    self.m21_ = np.zeros((2, 2), np.float64)

    # pair-2nd
    self.m221_ = np.zeros((2, 2), np.float64)
    self.m222_ = np.zeros((2, 2), np.float64)

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
        self.qt_[i, j, k, l, m, n, o, p, q] = qt_[i, j, k, l, m, n, o, p, q]

        # m61_
        self.m61_[q, o, j, i, k, n] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m51_
        self.m51_[i, j, k, l, m] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m41_
        self.m41_[i, j, n, p] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m42_
        self.m42_[n, o, p, q] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m31_
        self.m311_[i, j, k] += self.qt_[i, j, k, l, m, n, o, p, q]
        self.m312_[i, n, k] += self.qt_[i, j, k, l, m, n, o, p, q]
        self.m313_[q, n, o] += self.qt_[i, j, k, l, m, n, o, p, q]
        self.m314_[n, o, p] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m21_
        self.m21_[j, i] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m22_
        self.m221_[i, k] += self.qt_[i, j, k, l, m, n, o, p, q]
        self.m222_[n, p] += self.qt_[i, j, k, l, m, n, o, p, q]

        # x_
        self.x_[i] += self.qt_[i, j, k, l, m, n, o, p, q]
        it.iternext()

    print('  chker: {:0<8.6f},   condition: {:0<8.2g},   x1: {:0<8.4f},  eta_sum:  {:0<8.4f}'
          .format(self.checker, self.condition, self.x_[1], eta_sum))

    # it = np.nditer(self.m22_, flags=['multi_index'])
    # while not it.finished:
    #     i, j = it.multi_index
    #     print('  self.m22_{}:  {:0<8.8f}'
    #           .format(it.multi_index, self.m22_[i, j]))
    #     print('  m22_{}:       {:0<8.8f} \n'
    #           .format(it.multi_index, m22_[i, j]))
    #     it.iternext()
