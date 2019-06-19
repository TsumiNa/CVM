#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from collections import defaultdict, namedtuple
from copy import deepcopy

from typing import Union, Callable
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from .normalizer import Normalizer
from .utils import UnitConvert
from .vibration import ClusterVibration


class Sample(defaultdict):
    """
    Sample is a object which store a group of configuration data for calculation
    """

    def __init__(
            self,
            label: str,
            *,
            tag: str = None,
            temperature: list = None,
            energies: pd.DataFrame = None,
            clusters: dict = None,
            mean: str = 'arithmetic',
            vibration: bool = True,
            skip: bool = False,
            x_1: float = 0.001,
            condition: float = 1e-07,
            host: str = 'host',
            r_0: Union[float, dict] = None,
            normalizer: Union[dict, Normalizer] = None,
            patch=None,
    ):
        super().__init__()
        self.label = label
        self.tag = tag
        self.mean = mean
        self.vibration = vibration
        self.condition = condition
        self.skip = skip
        self.x_1 = x_1
        self.patch = patch

        # ##########
        # private vars
        # ##########
        self._host = host
        self._r_0 = r_0
        self._ens = None
        self._lattice_func = None
        self._int_func = None
        self._clusters = None
        self._normalizer = None
        self._temp = None

        if normalizer is not None:
            self.set_normalizer(normalizer)
        if temperature is not None:
            self.set_temperature(temperature)
        if clusters is not None:
            self.set_clusters(**clusters)
        if energies is not None:
            self.set_energies(energies)

    def set_energies(self, energies):

        def _nest(_f):
            f_ = _f
            return lambda self_: self_[f_]

        if isinstance(energies, pd.DataFrame):
            self._ens = energies

            # calculate debye function
            energy_shift = energies[self._host]
            xs = UnitConvert.lc2ad(energies.index.values)
            energies = energies.drop(columns=[self._host])

            for c in energies:
                ys = energies[c]
                self[c] = ClusterVibration(c,
                                           xs,
                                           ys,
                                           energy_shift=energy_shift,
                                           mean=self.mean,
                                           vibration=self.vibration)
                setattr(self.__class__, c, property(_nest(c)))
                if self._normalizer and c in self._normalizer:
                    ys = energies[c] + self._normalizer[c]
                    c = f'{c}_'
                    self[c] = ClusterVibration(c,
                                               xs,
                                               ys,
                                               energy_shift=energy_shift,
                                               mean=self.mean,
                                               vibration=self.vibration)
                    setattr(self.__class__, c, property(_nest(c)))

        else:
            raise TypeError('energies must be <pd.DataFrame> but got %s' %
                            energies.__class__.__name__)

    def set_temperature(self, temp):
        l = len(temp)  # get length of 'temp'
        if l == 1:
            self._temp = np.array(temp, np.single)
        elif l == 3:
            self._temp = np.linspace(temp[0], temp[1], temp[2])
        else:
            raise NameError('temperature was configured in error format')

    @property
    def energies(self):
        return self._ens

    @property
    def clusters(self):
        return deepcopy(self._clusters)

    def set_clusters(self, **val):
        self._clusters = deepcopy(val)

    @property
    def normalizer(self):
        return self._normalizer

    def set_normalizer(self, val):

        def _nest(_f):
            f_ = _f
            return lambda self_: self_[f_]

        if isinstance(val, Normalizer):
            pass
        elif isinstance(val, dict):
            val = Normalizer(**val)
        else:
            raise TypeError('normalizer must be a dict or has type of <Normalizer> but got %s' %
                            val.__class__.__name__)

        self._normalizer = val
        if self._ens is not None:
            energy_shift = self._ens[self._host]
            xs = UnitConvert.lc2ad(self._ens.index.values)
            for k, v in self._normalizer.items():
                if k in self._ens:
                    ys = (self._ens[k] + v)
                    k = f'{k}_'
                    self[k] = ClusterVibration(k,
                                               xs,
                                               ys,
                                               energy_shift=energy_shift,
                                               mean=self.mean,
                                               vibration=self.vibration)
                    setattr(self.__class__, k, property(_nest(k)))

    def __call__(self,
                 T: float,
                 *,
                 r: float = None,
                 convert_r: bool = False,
                 vibration: bool = None,
                 energy_patch: Callable[[float, float], namedtuple] = None,
                 **kwargs):
        """Get interaction energies at given T, and r.
        
        Parameters
        ----------
        T : float
            Temperature.
        r : float, optional
            Atomic distance. By default ``None``.
        vibration: bool
            Specific whether or not to import the thermal vibration effect.
        convert_r: bool, optional
            If ``True``, convert parameter <r> to atomic distance.
        energy_patch: Callable[[float, float], namedtuple], optional
            A patch that will be used to correct the returns of interaction energy.
            By default ``None``.
        
        Returns
        -------
        namedtuple
            Named tuple contains calculated interaction energies.
        """

        del kwargs

        def _int(cluster):
            ret_ = 0
            for k, v in cluster.items():
                ret_ += self[k](T, r=r, vibration=vibration) * v
            return ret_

        if convert_r:
            r = UnitConvert.lc2ad(r)

        ret = {}
        for k, v in self._clusters.items():
            try:
                ret[k] = _int(v)
            except KeyError as e:
                raise KeyError(f'configuration of `{k}` in parameter <series.clusters> '
                               f'reference an unknown phase {e}')

        if energy_patch is not None:
            patch = energy_patch(T, r)

            for k, v in patch.items():
                ret[k] += v

        return namedtuple('interaction_energy', self._clusters.keys())(**ret)

    def ite(self, *, temperature: list = None, k: int = 3, vibration: bool = None, **kwargs):
        """Iterate over each temperature

        Parameters
        ----------
        temperature : list, optional
            Reset temperature steps, by default None
        k: int
            Degree of the smoothing spline. Must be <= 5. Default is k=3, a cubic spline. 
        vibration: bool
            Specific whether or not to import the thermal vibration effect.

        Yields
        -------
        T: float
            Temperature at current step.
        r_func: Callable[[float], float]
            A function receiving impurity concentration c and
            returns corresponding atomic distance r.
        """
        del kwargs

        def r_0_func(t):
            x_mins = []
            c_mins = []

            for k_, v in self._r_0.items():
                _, x_min = self[k_](t, min_x='ws', vibration=vibration)
                x_mins.append(x_min)
                c_mins.append(v)

            tmp = np.array([c_mins, x_mins])
            index = np.argsort(tmp[0])
            return UnivariateSpline(tmp[0, index], tmp[1, index], k=k)

        if temperature is not None:
            self.set_temperature(temperature)

        for t in self._temp:
            if isinstance(self._r_0, dict):
                yield t, r_0_func(t)
            else:
                yield t, lambda _: self._r_0

    def __repr__(self):
        s1 = '  | \n  |-'
        s2 = '  | '
        if self.tag is not None:
            header = [f'{self.tag}-<{self.label}>-<skip: {self.skip}>:']
        else:
            header = [f'{self.label}-<{self.skip}>:']

        return f'\n{s1}'.join(header + [f'\n{s2}'.join(str(self.normalizer).split('\n'))] +
                              [f'\n{s2}'.join(str(v).split('\n')) for v in self.values()])
