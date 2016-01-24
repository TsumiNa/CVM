#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def __eta(self, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14):
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
                 self.enFP[p1, p2, p3, p4, p5, p6, p7, p8,
                           p9, p10, p11, p12, p13, p14] +
                 (self.beta / 14) *
                 (self.mu[p1] + self.mu[p2] + self.mu[p3] +
                  self.mu[p4] + self.mu[p5] + self.mu[p6] +
                  self.mu[p7] + self.mu[p8] + self.mu[p9] +
                  self.mu[p10] + self.mu[p11] + self.mu[p12] +
                  self.mu[p13] + self.mu[p14]))

    # m61
    m61 = self.m61_[p3, p1, p2, p6, p7, p9] * \
        self.m61_[p4, p5, p2, p8, p7, p9] * \
        self.m61_[p13, p14, p11, p8, p7, p9] * \
        self.m61_[p10, p12, p11, p6, p7, p9] * \
        self.m61_[p1, p5, p2, p9, p6, p8] * \
        self.m61_[p3, p4, p2, p7, p6, p8] * \
        self.m61_[p12, p13, p11, p7, p6, p8] * \
        self.m61_[p10, p14, p11, p9, p6, p8] * \
        self.m61_[p1, p10, p9, p6, p2, p11] * \
        self.m61_[p3, p12, p6, p7, p2, p11] * \
        self.m61_[p4, p13, p7, p8, p2, p11] * \
        self.m61_[p5, p14, p8, p9, p2, p11]

    # m51
    m51 = self.m51_[p1, p3, p4, p5, p2] * self.m51_[p10, p12, p13, p14, p11] *\
        self.m51_[p1, p3, p12, p10, p6] * self.m51_[p5, p4, p13, p14, p8] * \
        self.m51_[p1, p5, p14, p10, p9] * self.m51_[p3, p4, p13, p12, p7]

    # m41
    m41 = self.m41_[p1, p2, p6, p9] * self.m41_[p3, p2, p7, p6] * \
        self.m41_[p4, p2, p8, p7] * self.m41_[p5, p2, p9, p8] * \
        self.m41_[p10, p11, p6, p9] * self.m41_[p12, p11, p7, p6] * \
        self.m41_[p13, p11, p8, p7] * self.m41_[p14, p11, p9, p8]

    # m31
    m31 = self.m31_[p1, p2, p3] * self.m31_[p1, p6, p3] * \
        self.m31_[p3, p2, p4] * self.m31_[p3, p7, p4] * \
        self.m31_[p4, p2, p5] * self.m31_[p4, p8, p5] * \
        self.m31_[p5, p2, p1] * self.m31_[p5, p9, p1] * \
        self.m31_[p1, p6, p10] * self.m31_[p1, p9, p10] * \
        self.m31_[p3, p6, p12] * self.m31_[p3, p7, p12] * \
        self.m31_[p4, p7, p13] * self.m31_[p4, p8, p13] * \
        self.m31_[p5, p8, p14] * self.m31_[p5, p9, p14] * \
        self.m31_[p10, p11, p12] * self.m31_[p10, p6, p12] * \
        self.m31_[p12, p11, p13] * self.m31_[p12, p7, p13] * \
        self.m31_[p13, p11, p14] * self.m31_[p13, p8, p14] * \
        self.m31_[p14, p11, p10] * self.m31_[p14, p9, p10] * \
        self.m31_[p11, p6, p2] * self.m31_[p11, p7, p2] * \
        self.m31_[p11, p8, p2] * self.m31_[p11, p9, p2] * \
        self.m31_[p6, p2, p8] * self.m31_[p6, p7, p8] * \
        self.m31_[p6, p11, p8] * self.m31_[p6, p9, p8] * \
        self.m31_[p7, p11, p9] * self.m31_[p7, p8, p9] * \
        self.m31_[p7, p2, p9] * self.m31_[p7, p6, p9]

    # m21
    m21 = self.m21_[p1, p2] * self.m21_[p1, p6] * self.m21_[p1, p9] * \
        self.m21_[p3, p2] * self.m21_[p3, p6] * self.m21_[p3, p7] * \
        self.m21_[p4, p2] * self.m21_[p4, p7] * self.m21_[p4, p8] * \
        self.m21_[p5, p2] * self.m21_[p5, p8] * self.m21_[p5, p9] * \
        self.m21_[p10, p6] * self.m21_[p10, p9] * self.m21_[p10, p11] * \
        self.m21_[p12, p6] * self.m21_[p12, p7] * self.m21_[p12, p11] * \
        self.m21_[p13, p7] * self.m21_[p13, p8] * self.m21_[p13, p11] * \
        self.m21_[p14, p8] * self.m21_[p14, p9] * self.m21_[p14, p11] * \
        self.m21_[p2, p6] * self.m21_[p2, p9] * self.m21_[p6, p9] * \
        self.m21_[p2, p7] * self.m21_[p2, p8] * self.m21_[p7, p8] * \
        self.m21_[p11, p6] * self.m21_[p11, p7] * self.m21_[p6, p7] * \
        self.m21_[p11, p8] * self.m21_[p11, p9] * self.m21_[p8, p9]

    # x
    x = self.x_[p1] * self.x_[p2] * self.x_[p3] * \
        self.x_[p4] * self.x_[p5] * self.x_[p6] * \
        self.x_[p7] * self.x_[p8] * self.x_[p9] * \
        self.x_[p10] * self.x_[p11] * self.x_[p12] * \
        self.x_[p13] * self.x_[p14]

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
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14 = \
            it.multi_index

        fp_[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14] = \
            __eta(self, p1, p2, p3, p4, p5, p6, p7, p8,
                  p9, p10, p11, p12, p13, p14)

        eta_sum += fp_[p1, p2, p3, p4, p5, p6, p7, p8,
                       p9, p10, p11, p12, p13, p14]
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
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14 = \
            it.multi_index

        fp_[p1, p2, p3, p4, p5, p6, p7, p8,
            p9, p10, p11, p12, p13, p14] /= eta_sum

        self.checker += \
            np.absolute(fp_[p1, p2, p3, p4, p5, p6, p7, p8,
                            p9, p10, p11, p12, p13, p14] -
                        self.fp_[p1, p2, p3, p4, p5, p6, p7, p8,
                                 p9, p10, p11, p12, p13, p14])

        # fp_
        self.fp_[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14] =\
            fp_[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]

        # m61_
        self.m61_[p3, p1, p2, p6, p7, p9] += \
            fp_[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]

        # m51_
        self.m51_[p1, p3, p4, p5, p2] += \
            fp_[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]

        # m41_
        self.m41_[p1, p2, p6, p9] += \
            fp_[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]

        # m31_
        self.m31_[p1, p2, p3] += \
            fp_[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]

        # m21_
        self.m21_[p1, p2] += \
            fp_[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]

        # x_
        self.x_[p1] += \
            fp_[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]
        it.iternext()

    print('  chker: {:0<8.6f},   condition: {:0<8.2g},   x1: {:0<8.4f},  eta_sum:  {:0<8.4f}'
          .format(self.checker, self.condition, self.x_[1], eta_sum))
