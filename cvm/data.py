#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
CVM data class
==============

This is a data class for data storage and share
"""


class data(object):

    """data storage class"""

    __slots__ = ('inp',  # INCAR
                 'output'  # output data
                 )

    def __init__(self, inp):
        super(data, self).__init__()
        self.inp = inp
        self.output = {}
