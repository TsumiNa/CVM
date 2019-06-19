#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import datetime as dt
from typing import List
from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple

import numpy as np

from .sample import Sample
from .utils import parse_input_set


class BaseCVM(defaultdict, metaclass=ABCMeta):
    """
    CVM is a general CVM (Cluster Variation Method) calculation framework using  NIM (Natural Iteration Method). 
 
    """

    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset all conditions for new calculation.
        """

    @abstractmethod
    def update_energy(self, e_ints, **kwargs):
        """
        Update energies.
        """

    @abstractmethod
    def process(self, **kwargs):
        """
        The main loop to execute a NIM calculation step-by-step..
        """

    def __init__(self, meta: dict, *, series: List[dict] = None, experiment: dict = None):
        """BaseCVM
        
        Parameters
        ----------
        meta : dict
            Meta information.
            It can be anything that will be used to describe this calculation set.
            This information will be passed over the whole calculation.
        series : List[dict], optional
            Sample data contains raw energies, and other calculation needed parameters, by default None
        experiment : dict, optional
            The experiment data, by default None
        
        """
        super().__init__()
        self.count = 0
        self.checker = np.float64(1.0)
        self.beta = None

        if not isinstance(meta, dict):
            raise TypeError('meta information must be a dict')

        meta = {k.lower(): v.lower() for k, v in meta.items()}
        meta['timestamp'] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.meta = meta

        self.experiment = experiment

        ##################
        # init series
        ##################
        for s in series:
            tmp = Sample(**s)
            self.add_sample(tmp)

    def add_sample(self, val: Sample):
        """Add sample data
        
        Parameters
        ----------
        val : Sample
            Instance of :py:class:`Sample`.
        
        """

        def _nest(_f):
            f_ = _f
            return lambda self_: self_[f_]

        if not isinstance(val, Sample):
            raise TypeError('sample must be a Sample instance')
        self[val.label] = val

        if val.tag is not None:
            setattr(self.__class__, f'tag_{val.tag}', property(_nest(val.label)))

    @classmethod
    def from_samples(cls, meta: dict, *samples: Sample, experiment: dict = None):
        """Build a CVM instance from series samples.
        
        Parameters
        ----------
        meta : dict
            Meta information.
            It can be anything that will be used to describe this calculation set.
            This information will be passed over the whole calculation.
        samples : Sample
            List of Series sample.
        experiment : dict, optional
            The experiment data, by default None

        Returns
        -------
        BaseCVM
            New CVM instance.
        """
        ret = cls(meta, experiment=experiment)
        for s in samples:
            ret.add_sample(s)
        return ret

    @classmethod
    def from_input_set(cls, path_of_set: str):
        """Build a CVM instance from input set.
        
        Parameters
        ----------
        path_of_set : str
            The path of input set dir.
        
        Returns
        -------
        BaseCVM
            New CVM instance.
        """
        inp = parse_input_set(path_of_set)
        return cls(**inp)

    def __call__(self,
                 *labels: str,
                 verbose: bool = False,
                 sample_paras: dict = {},
                 reset_paras: dict = {},
                 update_en_paras: dict = {},
                 process_paras: dict = {}):
        """
        Run the calculation.
        
        Parameters
        ----------
        labels: str
            If not ``None``, only execute listed series samples.
        verbose : bool, optional
            Set to ``True`` to show extra information when running calculations, by default False
        sample_ite_paras : dict, optional
            The parameters will be passed to ``sample.ite`` and ``sample.__call__`` method, by default empty.        
        reset_paras : dict, optional
            The parameters will be passed to ``self.reset`` method, by default empty.
        update_en_paras : dict, optional
            The parameters will be passed to ``self.update_energy`` method, by default empty.
        process_paras : dict, optional
            The parameters will be passed to ``self.process`` method, by default empty.
        """

        ret = namedtuple('status', 'label, temperature, concentration, num_of_ite, int_energy')

        # temperature iteration
        for label, sample in self.items():
            if sample.skip:
                continue
            if labels and label not in labels:
                continue

            self.x_[1] = sample.x_1
            self.x_[0] = 1 - sample.x_1

            for T, r_0 in sample.ite(**sample_paras):

                # β = 1/kt
                self.beta = np.power(8.6173303e-5 * T, -1)

                # reset
                self.count = 0
                self.checker = np.float64(1.0)
                self.reset(**reset_paras)
                while self.checker > sample.condition:
                    e_int = sample(T, r=r_0(self.x_[1]), **sample_paras)
                    self.update_energy(e_int, **update_en_paras)
                    self.process(**process_paras)

                    if verbose:
                        yield ret(label=label,
                                  temperature=T,
                                  concentration=self.x_[1],
                                  num_of_ite=self.count,
                                  int_energy=e_int)

                if not verbose:
                    yield ret(label=label,
                              temperature=T,
                              concentration=self.x_[1],
                              num_of_ite=self.count,
                              int_energy=e_int)

    def __repr__(self):
        s1 = '  | \n  |-'
        s2 = '  | '
        header = [self.__class__.__name__ + ':']

        return f'\n{s1}'.join(header + [f'\n{s2}'.join(str(v).split('\n')) for v in self.values()])
