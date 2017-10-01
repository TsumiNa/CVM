#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd


def process(opt):
    prefix_name = "_".join(
        (opt['meta']['prefix'], opt['meta']['host'], opt['meta']['impurity'],
         opt['meta']['suffix'])).lstrip('_')

    for i, res in enumerate(opt['results']):
        filename = "_".join((prefix_name, res['label']))
        df = pd.DataFrame(
            np.array((res['temp'], res['c'])),
            index=['T', 'c'],
            dtype='float64')
        df.to_csv(filename + '.txt')
