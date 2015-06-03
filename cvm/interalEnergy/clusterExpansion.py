#!/usr/bin/env python
# -*- coding:utf-8 -*-

# from .. import data


class process(object):

    """docstring for process"""

    def __init__(self, d):
        super(process, self).__init__()
        self.data = d
        self.data.output['after'] = self.data.inp['energy']*10
