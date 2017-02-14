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
def eV2Kbar(p):
    return p * 2.9421912e13 * 1e-8 / 27.21138505


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

    print("c1: {:f},  c2: {:f},  lambda: {:f}".format(c1, c2, lmd))
    print("r0: {:f},  x0: {:f}".format(r0, x0))
    print("Gruneisen constant: {:f}".format(gamma_0))
    print("Equilibrium lattice constant: {:f} a.u.".format(ad2lc(r0)))
    print("Bulk Modulus: {:f} Kbar".format(bulk_moduli))
    print("Debye temperature: {:f} K\n\n".format(debye_temp_0))


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
            bounds=([3, 3, 1, xs[0]], [8.5, 8.5, 2, xs[-1]])
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
        D_0 = np.float64(41.63516) * np.power(r0 * B / M, 1 / 2)
        return D_0, lambda r: D_0 * np.power(r0 / r, 3 * lmd * r / 2)

    c1, c2, lmd, r0, morse_potential = _morse_gene(xdata, ydata)

    x0 = np.exp(-lmd * r0)
    B_0 = - (c2 * (lmd**3)) / (6 * np.pi * np.log(x0))
    gamma_0 = lmd * r0 / 2
    debye_temp_0, debye_temp_func = _debye_temp_gene(
        r0, lmd, eV2Kbar(B_0), np.float(M))

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
        bulk_moduli=eV2Kbar(B_0),
        debye_temperature_0=debye_temp_0,
        debye_temperature_func=debye_temp_func,
        debye_func=lambda r, T: _debye_func(debye_temp_func(r) / T),
        debye_func_0=lambda T: _debye_func(debye_temp_0 / T),
    )


if __name__ == '__main__':
    xdata = np.array([6.6, 6.8, 7, 7.1, 7.2, 7.3,
                      7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8])
    host = np.array([-10093.50087, -10093.56036, -10093.5962, -10093.60762, -10093.61555, -10093.62048, -10093.62287, -10093.62314, -10093.62163, -10093.61865, -10093.61445, -10093.60928, -10093.60331]) * 13.605698066
    ydata = host + np.array([-0.00002222, -0.00002086, -0.00002088, -0.00002141, -0.00002392, -0.00002683, -0.00003284, -0.00004102, -0.00005496, -0.00007731, -0.00011052, -0.00016161, -0.0002401]) * 13.605698066 / 4

    ydata_func = UnivariateSpline(xdata, ydata)
    ydata_min = minimize_scalar(ydata_func, bounds=(6.6, 8), method='bounded')
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
    morse = lambda r: ret['morse'](r) + ydata_min_y
    theta_D = ret['debye_temperature_func']
    D = ret['debye_func']

    print("c1: {:f},  c2: {:f},  lambda: {:f}".format(c1, c2, lmd))
    print("r0: {:f},  x0: {:f}".format(r0, x0))
    print("Gruneisen constant: {:f}".format(gamma_0))
    print("Equilibrium lattice constant: {:f} a.u.".format(ad2lc(r0)))
    print("Bulk Modulus: {:f} Kbar".format(bulk_moduli))
    print("")

    CONST_T = 800
    # morse potential
    xdata_morse = np.linspace(6.6, 8, 50)
    ydata_morse = [morse(r) for r in lc2ad(xdata_morse)]
    ydata_morse_min = minimize_scalar(
        lambda r: morse(r), bounds=(lc2ad(6.6), lc2ad(8)), method='bounded')
    ydata_morse_min_x = ydata_morse_min.x
    ydata_morse_min_y = ydata_morse_min.fun

    # vibration
    def free_en_vib(r, T, bzc=8.6173303E-5):
        return morse(r) + (9 / 8) * bzc * theta_D(r)\
            - bzc * T * (D(r, T) - 3 * np.log(1 - np.exp(-(theta_D(r) / T))))
    ydata_vib = [free_en_vib(r, CONST_T) for r in lc2ad(xdata_morse)]
    ydata_vib_min = minimize_scalar(
        lambda r: free_en_vib(r, CONST_T),
        bounds=(lc2ad(6.6), lc2ad(8)), method='bounded')
    ydata_vib_min_x = ydata_vib_min.x
    ydata_vib_min_y = ydata_vib_min.fun

    print("minimum from morse: <{:f}, {:f}>".
          format(ydata_morse_min_x, ydata_morse_min_y))
    print("minimum from vibration: <{:f}, {:f}>".
          format(ydata_vib_min_x, ydata_vib_min_y))
    print("")

    r_800 = 2.947958
    en_vib = free_en_vib(r_800, CONST_T)
    print("lattice constance at {:06.2f}K: {:f}".format(CONST_T, r_800))
    print("free energy with vibration: {:f}".format(en_vib))

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(xdata, ydata, 'x--', label='raw')
    plt.plot(xdata_morse, ydata_morse, 'o:', label='morse')
    for T in [400, 600, 800, 1000, 1200, 1400, 1600]:
        ydata_vib = [free_en_vib(r, T) for r in lc2ad(xdata_morse)]
        plt.plot(xdata_morse, ydata_vib, '^:', label='vibration(T=' + str(T) + '$K$)')
    plt.ylabel('total energy ($eV$)')
    plt.xlabel('lattice parameter ($a.u.$)')

    plt.legend()
    plt.savefig('total energy of Pd', dpi=300)
    plt.show()
