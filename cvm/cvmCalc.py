#!/usr/bin/env python3
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
import yaml
import sys
import os
import tempfile
import datetime as dt
import re as regex

from .A1 import tetrahedron as T
from .A1 import tetraOctahedron as TO
from .A1 import tetraSquare as TS
from .A1 import doubleTetrahedron as DT
from .A1 import quadrupleTetrahedron as QT

pattern = regex.compile(r"(/\*)+.+?(\*/)", regex.S)  # remove comment in json


class CvmCalc(object):

    """
    Configuration for CVM Calculation.
    ==================================

    Initialaztion
    -------------
    We use a INCAR as our calculation by inner the program we trade it as json.
    You can write a INCAR in JSON and and saved it as UTF-8 and pass the file
    to program using '-inp FileName'. Also you can pass a dictionary object
    contained you input to the first paramater directly.

    The output will be saved as output.json by default.However you can set the
    output file's name and path in the INCAR.

    You can format output by yourself even if you want to stream it to database
    directly. Just pass a backend function to the second paramater.
    """

    __slots__ = (
        'backend',
        'workerpool',
        'arg_dict',  # argvs will be reformatted as {'option': 'value'}
        'method_dict',  # store calculation methods as {'name': method}
    )

    def __init__(self, inp=None, backend=None):
        """
        keep None when you don't want to custom by yourself.
        """
        super(CvmCalc, self).__init__()
        self.workerpool = []
        self.arg_dict = {}
        self.method_dict = {}

        # init method
        self.method_dict['T'] = T
        self.method_dict['DT'] = DT
        self.method_dict['TO'] = TO
        self.method_dict['TS'] = TS
        self.method_dict['QT'] = QT

        # parse flags
        cwd = os.getcwd() + '/'
        cmd_folder = os.path.dirname(__file__)
        if cmd_folder not in sys.path:
            sys.path.append(cmd_folder)

        self.__initArg()

        if 'backend' not in self.arg_dict:
            self.backend = None if backend is None else backend
        else:
            backend_path = cwd + self.arg_dict['backend'][0]
            backend_path = backend_path[:-3]
            try:
                self.backend = __import__(self.arg_dict['backend'][0][:-3])
            except ImportError as e:
                raise e
        if inp is not None:
            self.__run(inp)
        else:
            if 'inp' not in self.arg_dict:
                raise NameError('need a INPCARD!')
            with open(cwd + self.arg_dict['inp'][0]) as f:
                _content = f.read()
                _content = pattern.sub('', _content)
            f = tempfile.TemporaryFile(mode='w+t')
            f.write(_content)
            f.seek(0)
            inp = json.load(f)
            f.close()
            self.__run(inp)

    def __run(self, inp):
        #  calculation run on separate thread
        if 'methods' not in inp:
            raise NameError('need specify  calculation methods')
        for method in inp['methods']:
            worker = self.method_dict[method](inp)
            worker.start()
            self.workerpool.append(worker)

        # wait all done
        for worker in self.workerpool:
            worker.join()

        # join results
        worker = self.workerpool[0]
        for other in self.workerpool[1:]:
            worker.output['results'].extend(other.output['results'])

        log_name = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(os.getcwd() + '/log/' + log_name)
        if not os.path.exists('log/'):
            os.makedirs('log/')

        if self.backend is not None:
            self.backend.process(worker.output)

        if 'opt' in self.arg_dict:
            if self.arg_dict['opt'][0] == 'json':
                with open(log_path + '.json', 'w') as f:
                    json.dump(worker.output, f, indent=2)
                return
        with open(log_path + '.yaml', 'w') as f:
            yaml.dump(worker.output, f, default_flow_style=False, indent=3)

    def __initArg(self):
        if len(sys.argv) > 1:

            _key = ''
            _value = []  # temp key, val pair
            for _temp in sys.argv[1:]:
                if bool('-' in _temp):
                    if _key is not '':
                        self.arg_dict[_key] = _value
                        _value = []
                    _key = _temp.lstrip('-')
                else:
                    _value.append(_temp)

            if _key is not '':
                self.arg_dict[_key] = _value
            else:
                self._useage()
            del _value
            del _key

    @classmethod
    def check(cls):
        print(cls.arg_dict)
        print('Done')
