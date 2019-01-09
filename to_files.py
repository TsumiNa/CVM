#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd


def process(opt):
    prefix_name = "_".join(
        (opt['meta']['prefix'], opt['meta']['host'], opt['meta']['impurity'],
         opt['meta']['suffix'])).lstrip('_')
    excel = pd.ExcelWriter(prefix_name + '.xlsx')

    for _, res in enumerate(opt['results']):
        # filename = "_".join((prefix_name, res['label']))
        sheet_name = res['label'].replace('$', '').replace('\\mathrm{',
                                                           '').replace(
                                                               '}', '')
        df = pd.DataFrame(
            np.array((res['temp'], res['c'])),
            index=['T', 'c'],
            dtype='float64')
        # df.to_csv(filename + '.txt')
        df.to_excel(excel, sheet_name=sheet_name)

    excel.save()
