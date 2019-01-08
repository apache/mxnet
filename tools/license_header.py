#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Add or check license header

Usuage:

- add the default license header to source files that do not contain a valid
  license:

  license_header.py add

- check if every files has a license header

  license_header.py check
"""

import re
import os
import argparse
from itertools import chain
import logging
import sys
import subprocess

# the default apache license
_LICENSE = """Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License."""

# if a file contains any str in the list, then consider it has been licensed
_LICENSE_PATTERNS = ['Licensed to the Apache Software Foundation']

# the folders or files that will be ignored
_WHITE_LIST = ['3rdparty',
               'docker/Dockerfiles',
               'example/image-classification/predict-cpp/image-classification-predict.cc',
               'example/rcnn/rcnn/cython/nms_kernel.cu',
               'example/rcnn/rcnn/cython/setup.py',
               'example/ssd/dataset/pycocotools/coco.py',
               'julia/REQUIRE',
               'prepare_mkl.sh',
               'R-package/',
               'scala-package/init-native/src/main/native/org_apache_mxnet_init_native_c_api.h',
               'scala-package/native/src/main/native/org_apache_mxnet_native_c_api.h',
               'src/operator/contrib/ctc_include/',
               'src/operator/contrib/nn/deformable_im2col.cuh',
               'src/operator/contrib/nn/deformable_im2col.h',
               'src/operator/contrib/psroi_pooling-inl.h',
               'src/operator/mkl/',
               'src/operator/nn/im2col.cuh',
               'src/operator/nn/im2col.h',
               'src/operator/nn/pool.h',
               'src/operator/special_functions-inl.h'
               ]

# language extensions and the according commment mark
_LANGS = {'.cc':'*', '.h':'*', '.cu':'*', '.cuh':'*', '.py':'#',
          '.pm':'#', '.scala':'*', '.cc':'*', '.sh':'#', '.cmake':'#',
          '.java':'*', '.sh':'#', '.cpp':'*', '.hpp':'*', '.c':'*',
          '.bat':'rem', '.pl':'#', '.m':'%', '.R':'#', '.mk':'#', '.cfg':'#',
          '.t':'#', '.ps1':'#', '.jl':'#'}

# Previous license header, which will be removed
_OLD_LICENSE = re.compile('.*Copyright.*by Contributors')


def get_mxnet_root():
    curpath = os.path.abspath(os.path.dirname(__file__))
    def is_mxnet_root(path: str) -> bool:
        return os.path.exists(os.path.join(path, ".mxnet_root"))
    while not is_mxnet_root(curpath):
        parent = os.path.abspath(os.path.join(curpath, os.pardir))
        if parent == curpath:
            raise RuntimeError("Got to the root and couldn't find a parent folder with .mxnet_root")
        curpath = parent
    return curpath


def _lines_have_license(lines):
    return any([any([p in l for p in _LICENSE_PATTERNS]) for l in lines])


def _get_license(comment_mark):
    if comment_mark == '*':
        body = '/*\n'
    else:
        body = ''
    for l in _LICENSE.split('\n'):
        if comment_mark == '*':
            body += ' '
        body += comment_mark
        if len(l):
            body += ' ' + l
        body += '\n'

    if comment_mark == '*':
        body += ' */\n'
    body += '\n'
    return body


def should_have_license(fname):
    if any([l in fname for l in _WHITE_LIST]):
        logging.debug('skip ' + fname + ', it matches the white list')
        return False
    _, ext = os.path.splitext(fname)
    if ext not in _LANGS:
        logging.debug('skip ' + fname + ', unknown file extension')
        return False
    return True


def file_has_license(fname):
    if not should_have_license(fname):
        return True
    try:
        with open(fname, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        if not lines or _lines_have_license(lines):
            return True
        else:
            logging.error("File %s doesn't have a license", fname)
            return False
    except UnicodeError:
        return True
    return True


def file_add_license(fname):
    if not should_have_license(fname):
        return
    with open(fname, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    if _lines_have_license(lines):
        return
    _, ext = os.path.splitext(fname)
    with open(fname, 'w', encoding="utf-8") as f:
        # shebang line
        if lines[0].startswith('#!'):
            f.write(lines[0].rstrip()+'\n\n')
            del lines[0]
        f.write(_get_license(_LANGS[ext]))
        for l in lines:
            f.write(l.rstrip()+'\n')
    logging.info('added license header to ' + fname)
    return


def under_git():
    return subprocess.run(['git', 'rev-parse', 'HEAD'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def git_files():
    return list(map(os.fsdecode,
        subprocess.check_output('git ls-tree -r master --name-only -z'.split()).split(b'\0')))


def file_generator(path: str):
    for (dirpath, dirnames, files) in os.walk(path):
        for file in files:
            yield os.path.abspath(os.path.join(dirpath, file))


def foreach(fn, iterable):
    for x in iterable:
        fn(x)


def script_name():
    """:returns: script name with leading paths removed"""
    return os.path.split(sys.argv[0])[1]


def main():
    logging.basicConfig(
        format='{}: %(levelname)s %(message)s'.format(script_name()),
        level=os.environ.get("LOGLEVEL", "INFO"))

    parser = argparse.ArgumentParser(
        description='Add or check source license header')

    parser.add_argument(
        'action', nargs=1, type=str,
        choices=['add', 'check'], default='add',
        help = 'add or check')

    parser.add_argument(
        'file', nargs='*', type=str, action='append',
        help='Files to add license header to')

    args = parser.parse_args()
    action = args.action[0]
    files = list(chain(*args.file))
    if not files and action =='check':
        if under_git():
            logging.info("Git detected: Using files under version control")
            files = git_files()
        else:
            logging.info("Using files under mxnet sources root")
            files = file_generator(get_mxnet_root())

    if action == 'check':
        if not all(map(file_has_license, files)):
            return 1
        else:
            logging.info("All known and whitelisted files have license")
            return 0
    else:
        assert action == 'add'
        foreach(file_add_license, files)
    return 0

if __name__ == '__main__':
    sys.exit(main())
