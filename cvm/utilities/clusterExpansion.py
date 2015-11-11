#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import numpy as np


class process(object):

    __slots__ = ('data')

    def __init__(self, data):
        super(process, self).__init__()
        self.__check(data)
        self.data = data
        self.__run()

    def __check(self, data):
        """
        something check for input data.
        """

        if 'int_en' not in data.inp:
            print('Need interaction energy')
            sys.exit(0)

    def __run(self):
        _int_pair_list = np.array(self.data.inp['int_en']['pair'])
        _int_trip_list = np.array(self.data.inp['int_en']['triple'])
        _int_tetr_list = np.array(self.data.inp['int_en']['tetrad'])
        _int_pair = self.__pair_int_generator(_int_pair_list)

        self.data.output['pair_int'] = _int_pair

    def __pair_int_generator(self, neighb, cut=None):

        # coordination number
        _coord_num = np.array([12,      # 1st
                               6,       # 2nd
                               24,      # 3rd
                               12,      # 4th
                               24,      # 5th
                               8,       # 6th
                               48,      # 7th
                               6,       # 8th
                               24,      # 9th
                               12,      # 10th
                               24,      # 11th
                               24,      # 12th
                               48,      # 13th
                               48,      # 14th
                               12,      # 15th
                               24,      # 16th
                               24,      # 17th
                               6,       # 17th
                               48,      # 18th
                               24,      # 19th
                               48])     # 20th

        # fix to input length
        _tmp = _coord_num[:neighb.size]

        # cut if needed
        if cut is not None:
            for x in cut:
                _tmp[x] = 0

        # calculation pair interaction
        _int_pair = 0.0
        _pair_list = []
        for index in range(len(_tmp)):
            _int_pair += _tmp[index]*neighb[index]/12
            print('pair interaction with %sth approximation is %s' %
                  (index+1, _int_pair))
            _pair_list.append(_int_pair)

        return _pair_list
