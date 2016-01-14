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
    m61 = qojnki * qojplm * npjokl * npjqmi
    m51 = ijklm
    m41 = jqni * jnok * jopl * jpqm
    m42 = nopq
    m31 = ikj * ikn * klj * klo * lmj * lmp * mij *
          miq * npo * npj * npq *qon * qoj * qop
    m21 = ji * jk * jl * jm * jn * jo * jp * jq * iq * in *
          kn * ko * lo * lp * mp * mq * nq * no * pq * po
    m22 = ik * im * lk * lm * np * oq
    x = i * j * k * l * m * n * o * p * q
    """
    # exp
    exp = np.exp(-self.beta * self.enQT[i, j, k, l, m, n, o, p, q] +
                 (self.beta / 54) *
                 (self.mu[i] + self.mu[j] + self.mu[k] +
                  self.mu[l] + self.mu[m] + self.mu[n] +
                  self.mu[o] + self.mu[p] + self.mu[q]))

    # m61
    m61 = self.m61_[q, o, j, i, k, n] * self.m61_[q, o, j, l, m, p] * \
        self.m61_[n, p, j, k, l, o] * self.m61_[n, p, j, m, i, q]

    # m51
    m51 = self.m51_[i, j, k, l, m]

    # m41
    m41 = self.m41_[i, j, n, p] * self.m41_[k, j, n, o] * \
        self.m41_[l, j, o, p] * self.m41_[m, j, q, p]

    # m42
    m42 = self.m42_[n, o, p, q]

    # m31
    m31 = self.m31_[i, k, j] * self.m31_[i, k, n] * self.m31_[k, l, j] * \
        self.m31_[k, l, o] * self.m31_[l, m, j] * self.m31_[l, m, p] * \
        self.m31_[m, i, j] * self.m31_[m, i, q] * self.m31_[n, p, o] * \
        self.m31_[n, p, j] * self.m31_[n, p, q] * self.m31_[q, o, n] * \
        self.m31_[q, o, j] * self.m31_[q, o, p]

    # m21
    m21 = self.m21_[j, i] * self.m21_[j, k] * self.m21_[j, l] * \
        self.m21_[j, m] * self.m21_[j, n] * self.m21_[j, o] * \
        self.m21_[j, p] * self.m21_[j, q] * self.m21_[i, q] * \
        self.m21_[i, n] * self.m21_[k, n] * self.m21_[k, o] * \
        self.m21_[l, o] * self.m21_[l, p] * self.m21_[m, p] * \
        self.m21_[m, q] * self.m21_[n, q] * self.m21_[n, o] * \
        self.m21_[p, q] * self.m21_[p, o]

    # m22
    m22 = self.m22_[i, k] * self.m22_[i, m] * self.m22_[l, k] * \
        self.m22_[l, m] * self.m22_[n, p] * self.m22_[o, q]

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
    self.m31_ = np.zeros((2, 2, 2), np.float64)

    # pair
    self.m21_ = np.zeros((2, 2), np.float64)
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
            (qt_[i, j, k, l, m, n, o, p, q] + 3 * self.qt_[i, j, k, l, m, n, o, p, q]) / 4

        # m61_
        self.m61_[q, o, j, i, k, n] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m51_
        self.m51_[i, j, k, l, m] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m41_
        self.m41_[i, j, n, p] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m42_
        self.m42_[n, o, p, q] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m31_
        self.m31_[i, k, j] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m21_
        self.m21_[j, i] += self.qt_[i, j, k, l, m, n, o, p, q]

        # m22_
        self.m22_[i, k] += self.qt_[i, j, k, l, m, n, o, p, q]

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
