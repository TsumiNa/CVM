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

import interalEnergy
import entropy
import naturalIteMethod

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

    __slots__ = ('backend')

    inp = {}  # input
    outp = {}  # output
    arg_dict = {}  # argvs will be reformatted as {'option': 'value'}

    def __init__(self, inp=None, backend=None):
        """
        keep None when you don't want to custom yourself.
        """
        CvmCalc._init_arg_dict()

        if inp is not None:
            CvmCalc.inp = inp
        else:
            if 'inp' not in CvmCalc.arg_dict:
                print('Need a INCAR!')
                sys.exit(0)
            with open(os.getcwd() + '/' + CvmCalc.arg_dict['inp'][0]) as f:
                CvmCalc.inp = json.load(f)

        if backend is not None:
            self.backend = backend

    def run():
        pass

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
        print(type(self.inp))
