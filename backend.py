#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def draw(opt):
    plt.figure(figsize=(8, 5), dpi=200)  # 800 * 500
    # Y axis
    y_axis = opt['Meta']['Temperature']

    # x-axis
    res_size = len(opt['Results'])
    data_size = len(opt['Results'][0]['data'])
    x_axis = np.zeros((res_size, data_size), np.float_)
    label = np.zeros((res_size), 'U10')
    for i in range(res_size):
        label[i] = 'μ= ' + '{:07.4}'.format(opt['Results'][i]['mu'])
        for j in range(data_size):
            x_axis[i][j] = opt['Results'][i]['data'][j]['c']

    for i in range(len(x_axis)):
        # plt.xlim(xmin=0.5, xmax=12.5)
        # plt.axhline(y=0, color='k', ls='-', lw=1.0)

        # draw
        plt.plot(x_axis[i], y_axis, '-', mew=1.5, mfc='w', ms=6, lw=1.5,
                 label=label[i])

    # for preview
    plt.grid(axis='y')
    plt.ylabel(r'Temperature ($K$)')
    plt.xlabel(r'Concentration of Rh ($\%$)')
    plt.legend(loc='lower right')
    plt.savefig(opt['Meta']['Name'], dpi=200)  # 200 dpi
    plt.show()
