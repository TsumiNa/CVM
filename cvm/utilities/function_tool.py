import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar
from scipy.interpolate import UnivariateSpline


# lattice constan to atomic distance
def lc2ad(d, n=4):
    return d * np.power((3 / (4 * n * np.pi)), 1 / 3)


# latomic distance to attice constan
def ad2lc(d, n=4):
    return d / np.power((3 / (4 * n * np.pi)), 1 / 3)


# a.u. press to Kbar
def au2Kbar(p):
    return p * 2.9421912e13 * 1e-8 / 2


# a.u. temperature to K
def au2K(t):
    return t * 3.1577464e5


def show_parameter(ret):
    c1 = ret['c1']
    c2 = ret['c2']
    lmd = ret['lmd']
    r0 = ret['r0']
    x0 = ret['x0']
    gamma_0 = ret['gamma_0']
    bulk_moduli = ret['bulk_moduli']

    debye_temp_0 = ret['debye_temperature_0']
    debye_300K = ret['debye_func_0'](1)

    print("c1: {:f},  c2: {:f},  lambda: {:f}".format(c1, c2, lmd))
    print("r0: {:f},  x0: {:f}".format(r0, x0))
    print("Gruneisen constant: {:f}".format(gamma_0))
    print("Equilibrium lattice constant: {:f} a.u.".format(ad2lc(r0)))
    print("Bulk Modulus: {:f} Kbar".format(au2Kbar(bulk_moduli)))
    print("Debye temperature: {:f} K".format(debye_temp_0))
    print("Debye at 300K: {:f}".format(debye_300K))


def thermal_vibration_parameters(xdata, ydata, M):
    # generate morse potential
    def _morse_gene(xs, ys):
        def morse_mod(r, c1, c2, lmd, r0):
            return c1 - 2 * c2 * np.exp(-lmd * (r - r0))\
                + c2 * np.exp(-2 * lmd * (r - r0))

        # morse parameters
        popt, err = curve_fit(
            morse_mod,
            xs,
            ys,
            bounds=([0, 0, 1, xs[0]], [1, 1, 2, xs[-1]])
        )

        return popt[0], popt[1], popt[2], popt[3],\
            lambda r: morse_mod(r, popt[0], popt[1], popt[2], popt[3])

    # debye function
    # default n=3
    def _debye_func(x, n=3):
        try:
            from scipy.integrate import quad
        except Exception as e:
            raise e

        ret, _ = quad(
            lambda t: t**n / (np.exp(t) - 1),
            0,
            x
        )
        return (n / x**n) * ret

    # generate debye temperature Θ_D
    def _debye_temp_gene(r0, lmd, B, M):  # (ΘD)0 r=r0
        D_0 = np.float(41.63516) * np.power(r0 * B / M, 1 / 2)
        return D_0, lambda r: D_0 * np.power(r0 / r, 3 * lmd * r / 2)

    c1, c2, lmd, r0, morse_potential = _morse_gene(xdata, ydata)

    x0 = np.exp(-lmd * r0)
    B_0 = - (c2 * (lmd**3)) / (6 * np.pi * np.log(x0))
    gamma_0 = lmd * r0 / 2
    debye_temp_0, debye_temp_func = _debye_temp_gene(
        r0, lmd, au2Kbar(B_0), np.float(M))

    # parameters will be used to construt
    # free energy with thermal vibration effect
    return dict(
        c1=c1,
        c2=c2,
        lmd=lmd,
        r0=r0,
        x0=x0,
        gamma_0=gamma_0,
        equilibrium_lattice_constant=ad2lc(r0),
        morse=morse_potential,
        bulk_moduli=B_0,
        debye_temperature_0=debye_temp_0,
        debye_temperature_func=debye_temp_func,
        debye_func=lambda r, T: _debye_func(debye_temp_func(r) / T),
        debye_func_0=lambda T: _debye_func(debye_temp_0 / T),
    )


if __name__ == '__main__':
    xdata = np.array([6.6, 6.8, 7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8])
    ydata = np.array([-7776.055853, -7776.102158, -7776.118432, -7776.118209, -7776.113401, -7776.10464, -7776.092511, -7776.077544, -7776.060214, -7776.040942, -7776.020108, -7775.998057, -7775.97508])

    ydata_func = UnivariateSpline(xdata, ydata)
    ydata_min = minimize_scalar(ydata_func, bounds=(6.6, 8), method='bounded')
    ydata_min_x = ydata_min.x
    ydata_min_y = float(ydata_min.fun)

    M_pd = 106.4
    xs = lc2ad(xdata)
    ys = ydata - ydata_min_y

    ret = thermal_vibration_parameters(xs, ys, M_pd)

    c1 = ret['c1']
    c2 = ret['c2']
    lmd = ret['lmd']
    r0 = ret['r0']
    x0 = ret['x0']
    gamma_0 = ret['gamma_0']
    bulk_moduli = ret['bulk_moduli']
    morse = ret['morse']

    debye_temp_0 = ret['debye_temperature_0']
    debye_300K = ret['debye_func_0'](1)

    print("c1: {:f},  c2: {:f},  lambda: {:f}".format(c1, c2, lmd))
    print("r0: {:f},  x0: {:f}".format(r0, x0))
    print("Gruneisen constant: {:f}".format(gamma_0))
    print("Equilibrium lattice constant: {:f} a.u.".format(ad2lc(r0)))
    print("Bulk Modulus: {:f} Kbar".format(au2Kbar(bulk_moduli)))
    print("Debye temperature: {:f} K".format(debye_temp_0))
    print("Debye at 300K: {:f}".format(debye_300K))

    xdata_morse = np.linspace(6.6, 8, 50)
    ydata_morse = [morse(lc2ad(r)) + ydata_min_y for r in xdata_morse]
    plt.plot(xdata_morse, ydata_morse, '^', label='morse')
    plt.plot(xdata_morse, ydata_func(xdata_morse), 'o', label='polynomial')
    plt.plot(xdata, ydata, 'x--')

    print("minimum at: {:f}".format(ydata_min_x))
    print("minimum is: {:f}".format(ydata_min_y))

    plt.show()
