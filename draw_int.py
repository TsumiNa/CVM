#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def percent(x, pos=0):
    return u'{:3.1f}'.format(100 * x)


def process(opt):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)  # 800 * 500
    plt.subplots_adjust(top=0.96, bottom=0.11, left=0.12, right=0.97)

    # draw results
    for res in opt['results']:
        # plt.xlim(xmin=0.5, xmax=12.5)
        # plt.axhline(y=0, color='k', ls='-', lw=1.0)
        # label[i] = 'int= ' + '{:07.4}'.format(opt['Results'][i]['1st_int'])
        ax.plot(res['temp'], res['1st'], 'o-', ms=3, lw=1.5,
                label='$\\tilde{E}_{int}^{1st}$ ' + res['label'][:-7])
        ax.plot(res['temp'], res['2nd'], 'o-', ms=3, lw=1.5,
                label='$\\tilde{E}_{int}^{2nd}$ ' + res['label'][:-7])
        # ax.plot(res['temp'], res['4th'], 'o-', ms=3, lw=1.5,
        #         label='$E_{int}^{4th}$')
        # ax.plot(res['temp'], res['9th_a'], 'o-', ms=3, lw=1.5,
        #         label='$E_{int}^{9th_a}$')
        # ax.plot(res['temp'], res['9th_b'], 'o-', ms=3, lw=1.5,
        #         label='$E_{int}^{9th_b}$')
        # ax.plot(res['temp'], res['10th'], 'o-', ms=3, lw=1.5,
        #         label='$E_{int}^{10th}$')
        # ax.plot(res['temp'], res['trip'], '^-', ms=3, lw=1.5,
        #         label='$E_{int}^{trip}$')
        # ax.plot(res['temp'], res['tetra'], '>-', ms=3, lw=1.5,
        #         label='$E_{int}^{tetra}$')

    # set formater
    # ax.xaxis.set_major_formatter(FuncFormatter(percent))
    # ax.set_xlim(0, 0.12)

    # for preview
    ax.grid(axis='y')
    # ax.set_ylim(400, 2000)
    ax.set_xlabel('Temperature ($K$)')
    ax.set_ylabel('Interaction energy ($eV$)')
    ax.legend(loc='center right', markerscale=1.0, fontsize=8)
    fig_name = "_".join((
        opt['meta']['prefix'],
        opt['meta']['host'],
        opt['meta']['impurity'],
        opt['meta']['suffix']
    )).lstrip('_') + ' int energy with temp depen'
    plt.savefig(fig_name, dpi=300)  # 300 dpi
    plt.show()
