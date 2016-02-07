#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def __eta(self, i, j, k, l, m, n, o, p, q, r, s, t, u, v):
    """
    η_p{1..14} = exp[-β*e_p{1..14} +
                    (β/14)(Σmu_p{1..14})]
                * m61^(1/2)
                * m51^(1/2)
                * m41^(-3/4)
                * m31^(-1/3)
                * m21^(1/3)
                * x^(-1/7)
    """
    # exp
    exp = np.exp(-self.beta *
                 self.enFP[i, j, k, l, m, n, o, p, q, r, s, t, u, v] +
                 (self.beta / 14) *
                 (self.mu[i] + self.mu[j] + self.mu[k] +
                  self.mu[l] + self.mu[m] + self.mu[n] +
                  self.mu[o] + self.mu[p] + self.mu[q] +
                  self.mu[r] + self.mu[s] + self.mu[t] +
                  self.mu[u] + self.mu[v]))

    # m61
    m61 = self.m61_[i, j, n, q, r, s] * \
        self.m61_[m, j, q, p, v, s] * \
        self.m61_[l, j, p, o, u, s] * \
        self.m61_[k, j, o, n, t, s] * \
        self.m61_[r, q, n, s, t, o] * \
        self.m61_[v, q, s, p, u, o] * \
        self.m61_[m, q, p, j, l, o] * \
        self.m61_[i, q, j, n, k, o] * \
        self.m61_[r, n, s, q, v, p] * \
        self.m61_[i, n, q, j, m, p] * \
        self.m61_[k, n, j, o, l, p] * \
        self.m61_[t, n, o, s, u, p]

    # m51
    m51 = self.m51_[i, j, k, l, m] * self.m51_[t, s, r, v, u] *\
        self.m51_[r, q, i, m, v] * self.m51_[k, o, t, u, l] * \
        self.m51_[r, n, t, k, i] * self.m51_[u, p, v, m, l]

    # m41
    m41 = self.m41_[i, j, n, q] * self.m41_[l, j, p, o] * \
        self.m41_[r, n, s, q] * self.m41_[v, s, p, q] * \
        self.m41_[t, s, n, o] * self.m41_[k, j, o, n] * \
        self.m41_[m, j, q, p] * self.m41_[u, s, o, p]

    # m31
    m31 = self.m31_[i, j, k] * self.m31_[t, s, r] * \
        self.m31_[l, o, k] * self.m31_[i, q, m] * \
        self.m31_[r, n, t] * self.m31_[u, p, v] * \
        self.m31_[m, j, i] * self.m31_[u, s, t] * \
        self.m31_[u, o, l] * self.m31_[r, q, i] * \
        self.m31_[i, n, r] * self.m31_[l, p, u] * \
        self.m31_[l, j, m] * self.m31_[v, s, u] * \
        self.m31_[t, o, u] * self.m31_[v, q, r] * \
        self.m31_[k, n, i] * self.m31_[m, p, l] * \
        self.m31_[k, j, l] * self.m31_[r, s, v] * \
        self.m31_[k, o, t] * self.m31_[m, q, v] * \
        self.m31_[t, n, k] * self.m31_[v, p, m] * \
        self.m31_[s, n, j] * self.m31_[s, p, j] * \
        self.m31_[o, n, q] * self.m31_[o, p, q] * \
        self.m31_[p, j, n] * self.m31_[p, s, n] * \
        self.m31_[s, o, j] * self.m31_[s, q, j] * \
        self.m31_[o, j, q] * self.m31_[o, s, q] * \
        self.m31_[p, o, n] * self.m31_[p, q, n]

    # m21
    m21 = self.m21_[r, s] * self.m21_[u, s] * self.m21_[t, o] * \
        self.m21_[l, o] * self.m21_[v, p] * self.m21_[u, p] * \
        self.m21_[v, s] * self.m21_[t, s] * self.m21_[u, o] * \
        self.m21_[k, o] * self.m21_[m, p] * self.m21_[l, p] * \
        self.m21_[i, j] * self.m21_[k, j] * self.m21_[r, q] * \
        self.m21_[v, q] * self.m21_[r, n] * self.m21_[t, n] * \
        self.m21_[l, j] * self.m21_[m, j] * self.m21_[m, q] * \
        self.m21_[i, q] * self.m21_[k, n] * self.m21_[i, n] * \
        self.m21_[n, o] * self.m21_[o, p] * self.m21_[n, s] * \
        self.m21_[s, p] * self.m21_[q, s] * self.m21_[s, o] * \
        self.m21_[p, q] * self.m21_[q, n] * self.m21_[p, j] * \
        self.m21_[j, n] * self.m21_[o, j] * self.m21_[j, q]

    # x
    x = self.x_[i] * self.x_[j] * self.x_[k] * \
        self.x_[l] * self.x_[m] * self.x_[n] * \
        self.x_[o] * self.x_[p] * self.x_[q] * \
        self.x_[r] * self.x_[s] * self.x_[t] * \
        self.x_[u] * self.x_[v]

    return (exp *
            np.power(m61, 1 / 2) *
            np.power(m51, 1 / 2) *
            np.power(m41, -3 / 4) *
            np.power(m31, -1 / 3) *
            np.power(m21, 1 / 3) *
            np.power(x, -1 / 7))


def process(self):
    # counts
    self.count += 1

    # calculate eta
    eta_sum = np.float64(0)
    fp_ = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2), np.float64)
    it = np.nditer(fp_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l, m, n, o, p, q, r, s, t, u, v = \
            it.multi_index

        fp_[i, j, k, l, m, n, o, p, q, r, s, t, u, v] = \
            __eta(self, i, j, k, l, m, n, o, p, q, r, s, t, u, v)

        eta_sum += fp_[i, j, k, l, m, n, o, p, q, r, s, t, u, v]
        it.iternext()

    ############################
    # normalization
    ############################
    self.checker = np.float64(0)

    # DT
    self.m61_ = np.zeros((2, 2, 2, 2, 2, 2), np.float64)

    # Square
    self.m51_ = np.zeros((2, 2, 2, 2, 2), np.float64)

    # pair
    self.m41_ = np.zeros((2, 2, 2, 2), np.float64)

    # 3-body
    self.m31_ = np.zeros((2, 2, 2), np.float64)

    # 1st-pair
    self.m21_ = np.zeros((2, 2), np.float64)

    # point
    self.x_ = np.zeros((2), np.float64)

    it = np.nditer(fp_, flags=['multi_index'])
    while not it.finished:
        i, j, k, l, m, n, o, p, q, r, s, t, u, v = \
            it.multi_index

        fp_[i, j, k, l, m, n, o, p, q, r, s, t, u, v] /= eta_sum

        self.checker += \
            np.absolute(fp_[i, j, k, l, m, n, o, p, q, r, s, t, u, v] -
                        self.fp_[i, j, k, l, m, n, o, p, q, r, s, t, u, v])

        # fp_
        self.fp_[i, j, k, l, m, n, o, p, q, r, s, t, u, v] =\
            fp_[i, j, k, l, m, n, o, p, q, r, s, t, u, v]

        # m61_
        self.m61_[i, j, n, q, r, s] += \
            fp_[i, j, k, l, m, n, o, p, q, r, s, t, u, v]

        # m51_
        self.m51_[i, j, k, l, m] += \
            fp_[i, j, k, l, m, n, o, p, q, r, s, t, u, v]

        # m41_
        self.m41_[i, j, n, q] += \
            fp_[i, j, k, l, m, n, o, p, q, r, s, t, u, v]

        # m31_
        self.m31_[i, j, k] += \
            fp_[i, j, k, l, m, n, o, p, q, r, s, t, u, v]

        # m21_
        self.m21_[r, s] += \
            fp_[i, j, k, l, m, n, o, p, q, r, s, t, u, v]

        # x_
        self.x_[i] += \
            fp_[i, j, k, l, m, n, o, p, q, r, s, t, u, v]
        it.iternext()

    print('  chker: {:0<8.6f},   condition: {:0<8.2g},   x1: {:0<8.4f},  eta_sum:  {:0<8.4f}'
          .format(self.checker, self.condition, self.x_[1], eta_sum))
