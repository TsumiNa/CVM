# import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def percent(x, pos=0):
    return u'{:3.1f}'.format(x / 7.4)


def process(opt):
    # config figure
    f, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
    plt.subplots_adjust(bottom=0.1, top=0.97, left=0.18, right=0.98)

    # horizontal axis for 3d,4sp element
    # element_axis = ['Sc', 'Ti', 'V', 'Cr', 'Mn',
    #                 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge']
    x_axis = [5.2325902, 7.4000000, 9.0631120, 10.4651804, 11.7004273, 12.8171760, 13.8441323, 14.8000000, 15.69777054, 15.69777054, 16.54690303]

    int_list = []

    t800 = [-0.156330696, 0.026852342, 0.004438451, 0.031264398, 0.007519461, -0.00283706, 0.003530679, -0.000524908,0.014907355, 0.001397849, 0.001059068]
    int_list.append((t800, 'without vib.'))

    for res in opt['results']:
        t800 = res['inter_en'][0]
        int_list.append((t800, '$T_{vib.}=800K$'))

        t1000 = res['inter_en'][1]
        int_list.append((t1000, '$T_{vib.}=1000K$'))

        t1200 = res['inter_en'][2]
        int_list.append((t1200, '$T_{vib.}=1200K$'))

    ax1.axhline(y=0, color='k', ls='-', lw=1.5)
    ax1.xaxis.set_major_formatter(FuncFormatter(percent))
    z = 5
    for int in int_list:
        ax1.plot(x_axis, int[0], 'o-', zorder=z,
                 mew=1.0, ms=6, lw=1.5, label=int[1]
                 )
        z -= 1
    # ax1.set_ylabel('', visible=False)

    # set xticks to element
    # plt.xticks(x_axis, element_axis)

    # use legend
    # ax = plt.gca()
    # plt.setp(ax.get_xmajorticklabels(), visible=True)
    plt.tick_params(labelsize=18)
    plt.legend(loc='lower right', fontsize=18)
    plt.annotate('$T_{FD}=800K$', (13.8, 0.03), fontsize=18)
    # plt.savefig('displacement of two impurities in Al.png', dpi=200)  # 150 dpi
    plt.figtext(0.02, 0.5, u'Interaction energy ($eV$)', size=20, horizontalalignment='center', verticalalignment='center', rotation='vertical')
    plt.figtext(0.5, 0.02, u'Distance ($a_0$)', size=20,
                horizontalalignment='center', verticalalignment='center')
    plt.savefig('ru-int-vib.png', dpi=300)  # 150 dpi
    plt.show()
