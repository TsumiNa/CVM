#!/usr/bin/env python3
# Copyright 2019 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import argparse

parser = argparse.ArgumentParser(prog='cvm', description=u"CVM calculation")
parser.add_argument("inp_card", help=u"input card for CVM calculation")
parser.add_argument(
    "-b", "--backend", nargs='*', help=u"set post processer to handle output results")
parser.add_argument("-o", "--output_json", action="store_true", help=u"format results to json")
args = parser.parse_args()
