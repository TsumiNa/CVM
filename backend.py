#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np


def percent(x, pos=0):
    return u'{:3.1f}'.format(100*x)


def draw(opt):
    fig = plt.figure(figsize=(8, 5), dpi=150)  # 800 * 500
    ax = fig.add_subplot(1, 1, 1)

    # Y axis
    y_axis = opt['Meta']['Temperature']

    # x-axis
    res_size = len(opt['Results'])
    data_size = len(opt['Results'][0]['data'])
    x_axis = np.zeros((res_size, data_size), np.float_)
    label = np.zeros((res_size), 'U11')
    for i in range(res_size):
        # label[i] = 'μ= ' + '{:07.4}'.format(opt['Results'][i]['mu'])
        label[i] = 'int= ' + '{:07.4}'.format(opt['Results'][i]['1st_int'])
        for j in range(data_size):
            x_axis[i][j] = opt['Results'][i]['data'][j]['c']

    for i in range(len(x_axis)):
        # plt.xlim(xmin=0.5, xmax=12.5)
        # plt.axhline(y=0, color='k', ls='-', lw=1.0)

        # draw
        ax.plot(x_axis[i], y_axis, '-', mew=1.5, mfc='w', ms=6, lw=1.5,
                label=label[i])

    y_exp = opt['Meta']['Experiment'][0]['temp']
    x_exp = opt['Meta']['Experiment'][0]['c']
    ax.plot(x_exp, y_exp, '--x', mew=1.5, mfc='w', ms=6, lw=1.5,
            label='Experiment')

    # set formater
    ax.xaxis.set_major_formatter(FuncFormatter(percent))

    # for preview
    ax.grid(axis='y')
    ax.set_ylabel(r'Temperature ($K$)')
    ax.set_xlabel(r'Concentration of Ru ($\%$)')
    ax.legend(loc='lower right')
    plt.savefig(opt['Meta']['Name'], dpi=200)  # 200 dpi
    plt.show()
