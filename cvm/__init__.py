#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
CVM
===

Provides
--------
combined cluster variation method with real spaces cluster expansion to
calculated solubility limit.
"""

import json
import sys
import os

from .data import data
from .interalEnergy import clusterExpansion as ce
from . import entropy as entropy
from . import naturalIteMethod as nim

__version__ = '0.0.1'


class CvmCalc(object):

    """
    Configuration for CVM Calculation.
    ==================================

    Initialaztion
    -------------
    Following general science calculation practice, we use a INCAR as our
    calculation by inner the program we trade it as json. You can write a
    INCAR in JSON and and saved it as UTF-8 and pass the file to program using
    '-i FileName'. Also you can pass a dictionary object contained you input to
    the first paramater directly.

    The output will be saved as output.json by default.However you can set the
    output file's name and path in the INCAR.

    You can format output by yourself even if you want to stream it to database
    directly. Just pass a backend function to the second paramater.
    """

    __slots__ = ('data', 'backend')

    arg_dict = {}  # argvs will be reformatted as {'option': 'value'}
    shared_data = {}  # share between subroutines

    def __init__(self, inp=None, backend=None):
        """
        keep None when you don't want to custom yourself.
        """
        super(CvmCalc, self).__init__()

        CvmCalc._init_arg_dict()
        if inp is not None:
            self.data = data(inp)
        else:
            if 'inp' not in CvmCalc.arg_dict:
                print('Need a INCAR!')
                sys.exit(0)
            with open(os.getcwd() + '/' + CvmCalc.arg_dict['inp'][0]) as f:
                self.data = data(json.load(f))

        self.backend = None if backend is None else backend

    def run(self):
        ce.process(self.data)

        if self.backend is not None:
            self.backend(self.data.output)
        else:
            with open(os.getcwd() + '/output.json', 'w') as f:
                json.dump(self.data.output, f)

    @classmethod
    def _init_arg_dict(cls):
        if len(sys.argv) > 1:

            _key = ''
            _value = []  # temp key, val pair
            for _temp in sys.argv[1:]:
                if bool('-' in _temp):
                    if _key is not '':
                        cls.arg_dict[_key] = _value
                        _value = []
                    _key = _temp.lstrip('-')
                else:
                    _value.append(_temp)

            if _key is not '':
                cls.arg_dict[_key] = _value
            else:
                cls._useage()
            del _value
            del _key

    @staticmethod
    def _usage():
        """
        Usage
        """
        pass

    def _check(self):
        print(CvmCalc.arg_dict)
