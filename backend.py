#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
# import numpy as np


def percent(x, pos=0):
    return u'{:3.1f}'.format(100 * x)


def draw(opt):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)  # 800 * 500

    # draw experiment
    y_exp = opt['Experiment'][0]['temp']
    x_exp = opt['Experiment'][0]['c']
    ax.plot(x_exp, y_exp, '--x', mew=1.5, mfc='w', ms=6, lw=1.5,
            label='Experiment')

    # draw results
    for res in opt['Results']:
        # plt.xlim(xmin=0.5, xmax=12.5)
        # plt.axhline(y=0, color='k', ls='-', lw=1.0)
        # label[i] = 'int= ' + '{:07.4}'.format(opt['Results'][i]['1st_int'])
        ax.plot(res['c'], res['temp'], '-', mew=1.5, mfc='w', ms=6, lw=1.5,
                label=res['label'])

    # set formater
    ax.xaxis.set_major_formatter(FuncFormatter(percent))

    # for preview
    ax.grid(axis='y')
    ax.set_ylabel(r'Temperature ($K$)')
    ax.set_xlabel(r'Concentration of Ru ($\%$)')
    ax.legend(loc='lower right')
    plt.savefig(opt['Meta']['Name'], dpi=200)  # 200 dpi
    plt.show()
