# Copyright 2019 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from collections import OrderedDict
import pandas as pd


class Results(OrderedDict):

    def add_result(self, label: str, results: list, *, tag: str = None):
        self[label] = pd.DataFrame(data=results)

        if tag is not None:
            setattr(self, tag, self[label])

    def __repr__(self):

        s1 = '  |-'
        header = [self.__class__.__name__ + ':']

        return f'\n{s1}'.join(header + [f'<{k}>' for k in self.keys()])
