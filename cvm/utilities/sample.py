#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


class Sample(object):
    """
    Sample is a object which store a group of configuration data for calculation
    """

    __slots__ = (
        'label',  # label for a calculation sample
        'x_1',  # initialization of impurity concentration
        'condition',  # Convergence condition
        'int',  # interaction energy
        'mu',  # Chemical potential
        'temp',  # Temperature (K)
        'transfer',
        'res',  # result
    )

    def __init__(self, label):
        super(Sample, self).__init__()
        self.label = label
        self.int = []
        self.res = {
            'label': label,
            'temp': [],
            'c': [],
            '1st': [],
            '2nd': [],
            '4th': [],
            'trip': [],
            'tetra': [],
        }

    @classmethod
    def effctive_en(cls, int_pair, args=None):
        """
        2nd parameter refer to the neighbour that transfer to
        """
        # coordination number
        to = args[0] if args else 1
        end = args[1] if args else 0
        start = args[2] if args else 2
        _coord_num = np.array([
            12,
            6,
            24,
            12,
            24,
            8,
            48,
            6,
            12,  # 9th-a
            24,  # 9th-b
            4,
            24,
            24,
            48,  # 13th-a
            24,  # 13th-b
            48,
            12,
            24,  # 16th-a
            24,  # 16th-b
            24,  # 17th-a
            6,   # 17th-b
            48,  # 18th-a
            24,  # 18th-b
            24,
            48  # 20th
        ])

        # prepare range
        length = len(int_pair)
        if start > length or end > length or start > end:
            raise IndexError('index error')
        if end == 0:
            _range = range(start - 1, length)
        else:
            _range = range(start - 1, end)

        # calculation pair interaction
        _int = np.float64(int_pair[to - 1])
        # print(_range)
        for index in _range:
            if index == to - 1:
                pass
            else:
                _int += _coord_num[index] * int_pair[index] /\
                    _coord_num[to - 1]
                # print('approximation until %sth is: %s eV' %
                #       (index + 1, _int))
        int_pair[to - 1] = _int

        return int_pair
