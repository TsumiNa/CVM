#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def percent(x, pos=0):
    return u'{:3.1f}'.format(100 * x)


def process(opt):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)  # 800 * 500
    plt.subplots_adjust(top=0.96, bottom=0.11, left=0.10, right=0.97)

    # draw experiment
    y_exp = opt['experiment'][0]['temp']
    x_exp = opt['experiment'][0]['c']
    ax.plot(x_exp, y_exp, 'x--', mew=3, mfc='w', ms=6, lw=1.5,
            label='Experiment')

    # theory
    # ax.plot(
    #     [0.0083, 0.019, 0.027, 0.035, .056], [800, 1200, 1400, 1600, 2050],
    #     'kx--', mew=3, mfc='w', ms=6, lw=1.5, label='Theory'
    # )

    # draw results
    for res in opt['results']:
        # plt.xlim(xmin=0.5, xmax=12.5)
        # plt.axhline(y=0, color='k', ls='-', lw=1.0)
        # label[i] = 'int= ' + '{:07.4}'.format(opt['Results'][i]['1st_int'])
        ax.plot(res['c'], res['temp'], 'o-', ms=4, lw=1.5,
                label=res['label'])

    # set formater
    ax.xaxis.set_major_formatter(FuncFormatter(percent))
    ax.set_xlim(0, 0.12)

    # for preview
    ax.grid(axis='y')
    # ax.set_ylim(400, 2000)
    ax.set_ylabel('Temperature ($K$)')
    ax.set_xlabel('Concentration of ' + opt['meta']['impurity'].capitalize() + '($\%$)')
    ax.legend(loc='lower right', markerscale=1.2, fontsize=13)
    fig_name = "_".join((
        opt['meta']['prefix'],
        opt['meta']['host'],
        opt['meta']['impurity'],
        opt['meta']['suffix']
    )).lstrip('_') + '.eps'
    plt.savefig(fig_name, dpi=300)  # 300 dpi
    plt.show()
