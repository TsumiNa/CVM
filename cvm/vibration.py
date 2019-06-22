from collections import Iterable

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit, minimize_scalar

from .utils import UnitConvert as uc
from .utils import mixed_atomic_weight


class ClusterVibration(object):
    """
    Tools class for cluster vibration
    """

    def __init__(self,
                 label,
                 xs,
                 ys,
                 mass,
                 num,
                 *,
                 energy_shift=None,
                 vibration=True,
                 morse_paras_bounds=None):

        if not len(xs) == len(ys):
            raise ValueError('xs and ys must have same dim.')
        self.vibration = vibration
        self.label = label
        if label[-1] == '_':
            label = label[:-1]
        self.mass, self.num = mass, num
        self._xs = self._check_input(xs)
        self._ys = self._check_input(ys) / num

        # parameter bounds for fitting
        # order: c1, c2, lmd, r0
        # bounds: ([low, low, low, low], [high, high, high, high])
        # doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        if morse_paras_bounds is not None:
            self._morse_paras_bounds = morse_paras_bounds
        else:
            self._morse_paras_bounds = ([0, 0, 0, xs[0]], [10, 10, 2, xs[-1]])

        # energy shift
        if energy_shift is not None:

            if isinstance(energy_shift, Iterable):
                if len(energy_shift) != len(ys):
                    raise ValueError('energy_shift must have same dim with ys')
                self._ys += np.array(energy_shift)
            else:
                self._ys += np.full_like(ys, energy_shift)
        self._shift = self._get_shift()
        self._ys -= self._shift

        # fit parameters
        self._paras = self._fit_paras()

        # calculate equilibrium constant
        poly_min = minimize_scalar(
            lambda _r: self.morse_potential(_r) + self._shift,
            bounds=(self._xs[0], self._xs[-1]),
            method='bounded')
        self._lattic_cons = poly_min.x
        self._ground_en = poly_min.fun

    def _get_shift(self):
        poly_min = minimize_scalar(
            UnivariateSpline(self._xs, self._ys, k=4),
            bounds=(self._xs[0], self._xs[-1]),
            method='bounded')

        return poly_min.fun

    def _check_input(self, array):
        if isinstance(array, list):
            return np.array(array, dtype=np.float64)
        if isinstance(array, pd.Series):
            return array.values
        if isinstance(array, np.ndarray):
            return array
        raise TypeError('input must be a array with shape (n,)')

    @property
    def c1(self) -> float:
        """Morse potential paramter `C1`.

        Returns
        -------
        float
        """
        return self._paras['c1']

    @property
    def c2(self) -> float:
        """Morse potential paramter `C2`.

        Returns
        -------
        float
        """
        return self._paras['c2']

    @property
    def lmd(self) -> float:
        """Morse potential paramter `lambda`.

        Returns
        -------
        float
        """
        return self._paras['lmd']

    @property
    def r_0(self) -> float:
        """Atomic distance at equilibrium status.

        Returns
        -------
        float
        """
        return self._paras['r_0']

    @property
    def equilibrium_lattice_cons(self) -> float:
        """Equilibrium lattice constant.

        Returns
        -------
        float
        """
        return uc.ad2lc(self._lattic_cons)

    @property
    def x_0(self) -> float:
        return self._paras['x_0']

    @property
    def gamma_0(self) -> float:
        return self._paras['gamma_0']

    @property
    def bulk_modulus(self) -> float:
        """Estimated bulk modulus.
        
        Returns
        -------
        float
        """
        return self._paras['B_0']

    def morse_potential(self, r: float) -> float:
        """Morse potential
        
        Parameters
        ----------
        r : float
            Atomic distance
        
        Returns
        -------
        float
            Potential energy.
        """
        return self.c1 - 2 * self.c2 * np.exp(-self.lmd * (r - self.r_0)) + \
            self.c2 * np.exp(-2 * self.lmd * (r - self.r_0))

    def debye_temperature(self, r: float = None) -> float:
        """Debye temperature function.
        
        Parameters
        ----------
        r : float, optional
            Atomic distance, by default None
        
        Returns
        -------
        float
        """
        D_0 = np.float64(41.63516) * np.power(self.r_0 * self.bulk_modulus / self.mass, 1 / 2)
        if r is None:
            return D_0
        return D_0 * np.power(self.r_0 / r, 3 * self.lmd * r / 2)

    # debye function
    # default n=3
    def debye_function(self, T: float, r: float = None, *, n: int = 3) -> float:
        """Debye function.
        
        Parameters
        ----------
        T : float
            Temperature.
        r : float, optional
            Atomic distance, by default None
        n : int, optional
            Dimssion, by default 3
        
        Returns
        -------
        float
        """
        x = self.debye_temperature(r) / T

        ret, _ = quad(lambda t: t**n / (np.exp(t) - 1), 0, x)
        return (n / x**n) * ret

    def __call__(self, T: float, *, r: float = None, min_x: str = None, vibration: bool = None):
        """Get free energy.

        Parameters
        ----------
        T : float
            Temperature.
        r : float, optional
            Atomic distance, by default ``None``.
        vibration: bool
            Specific whether or not to import the thermal vibration effect.
        min_x: str, optional
            By default ``None``.
            If not ``None``, function will returns equilibrium lattice constant as second result.
            The string can be ``ws`` or ``lattice``.
        
        """

        bzc = 8.6173303e-5
        if vibration is None:
            vibration = self.vibration

        if r is not None:
            if not vibration:
                return (self.morse_potential(r) + self._shift) * self._num

            # construct vibration withed energy formula
            return (self.morse_potential(r) + self._shift + \
                (9 / 8) * bzc * self.debye_temperature(r) - \
                bzc * T * (self.debye_function(T, r) - \
                3 * np.log(1 - np.exp(-(self.debye_temperature(r) / T))))) * self._num

        if not vibration:
            if min_x:
                if min_x == 'ws':
                    return self._ground_en * self._num, self._lattic_cons
                if min_x == 'lattice':
                    return self._ground_en * self._num, uc.ad2lc(self._lattic_cons)
                raise ValueError("min_x can only be 'ws' or 'lattice' but got %s" % min_x)
            return self._ground_en * self._num

        poly_min = minimize_scalar(
            lambda _r: self(T, r=_r, vibration=True),
            bounds=(self._xs[0], self._xs[-1]),
            method='bounded')

        if min_x:
            if min_x == 'ws':
                return poly_min.fun * self._num, poly_min.x
            if min_x == 'lattice':
                return poly_min.fun * self._num, uc.ad2lc(poly_min.x)
            raise ValueError("min_x can only be 'ws' or 'lattice' but got %s" % min_x)
        return poly_min.fun

    def _fit_paras(self):
        # morse potential
        def morse_pot(r, c1, c2, lmd, r0):
            return c1 - 2 * c2 * np.exp(-lmd * (r - r0))\
                + c2 * np.exp(-2 * lmd * (r - r0))

        # morse parameters
        popt, _ = curve_fit(morse_pot, self._xs, self._ys, bounds=self._morse_paras_bounds)
        c1, c2, lmd, r0 = popt[0], popt[1], popt[2], popt[3]

        x0 = np.exp(-lmd * r0)
        B0 = uc.eV2Kbar(-(c2 * (lmd**3)) / (6 * np.pi * np.log(x0)))
        gamma_0 = lmd * r0 / 2

        # parameters will be used to construt
        # free energy with thermal vibration effect
        return dict(
            c1=c1,
            c2=c2,
            lmd=lmd,
            r_0=r0,
            x_0=x0,
            gamma_0=gamma_0,
            B_0=B0,
        )

    def __repr__(self):
        s1 = '  |-'
        header = [f'{self.label}:']

        return f'\n{s1}'.join(header + [
            'c1: {:f},  c2: {:f},  lambda: {:f}'.format(self.c1, self.c2, self.lmd), \
            'r0: {:f},  x0: {:f}'.format(self.r_0, self.x_0), \
            'Gruneisen constant: {:f}'.format(self.gamma_0), \
            'Equilibrium lattice constant: {:f} a.u.'.format(self.equilibrium_lattice_cons), \
            'Bulk Modulus: {:f} Kbar'.format(self.bulk_modulus), \
            'Debye temperature: {:f} K'.format(self.debye_temperature())
        ])