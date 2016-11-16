#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse

from .cvmCalc import CvmCalc


parser = argparse.ArgumentParser(description=u"CVM calculation")
parser.add_argument("inp_card", action="store",
                    help=u"input card for CVM calculation"
                    )
parser.add_argument("-b", "--backend", action="store",
                    help=u"set post processer to handle output results"
                    )
parser.add_argument("-o", "--output_json", action="store_true", default=False,
                    help=u"format results to json"
                    )
args = parser.parse_args()

CvmCalc(**args.__dict__)
