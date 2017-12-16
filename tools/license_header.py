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

  python license_header.py add

- check if every files has a license header

  python license_header.py check
"""

import re
import os
import argparse

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
_WHITE_LIST = ['R-package/',
               'cub/',
               'dlpack/',
               'dmlc-core/',
               'mshadow/',
               'nnvm',
               '3rdparty',   
               'ps-lite',
               'src/operator/mkl/',
               'src/operator/contrib/ctc_include/']

# language extensions and the according commment mark
_LANGS = {'.cc':'*', '.h':'*', '.cu':'*', '.cuh':'*', '.py':'#',
          '.pm':'#', '.scala':'*', '.cc':'*', '.sh':'#', '.cmake':'#',
          '.java':'*', '.sh':'#', '.cpp':'*', '.hpp':'*', '.c':'*',
          '.bat':'rem', '.pl':'#'}

# Previous license header, which will be removed
_OLD_LICENSE = re.compile('.*Copyright.*by Contributors')

def _has_license(lines):
    return any([any([p in l.decode('utf-8') for p in _LICENSE_PATTERNS]) for l in lines])

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

def _valid_file(fname, verbose=False):
    if any([l in fname for l in _WHITE_LIST]):
        if verbose:
            print('skip ' + fname + ', it matches the white list')
        return False
    _, ext = os.path.splitext(fname)
    if ext not in _LANGS:
        if verbose:
            print('skip ' + fname + ', unknown file extension')
        return False
    return True

def process_file(fname, action, verbose=True):
    if not _valid_file(fname, verbose):
        return True
    with open(fname, 'rb') as f:
        lines = f.readlines()
    if not lines:
        return True
    if _has_license(lines):
        return True
    elif action == 'check':
        return False
    _, ext = os.path.splitext(fname)
    with open(fname, 'wb') as f:
        # shebang line
        if lines[0].startswith(b'#!'):
            f.write(lines[0].rstrip()+b'\n\n')
            del lines[0]
        f.write(str.encode(_get_license(_LANGS[ext])))
        for l in lines:
            f.write(l.rstrip()+b'\n')
    print('added license header to ' + fname)
    return False

def process_folder(root, action):
    excepts = []
    for root, _, files in os.walk(root):
        for f in files:
            fname = os.path.normpath(os.path.join(root, f))
            if not process_file(fname, action):
                excepts.append(fname)
    if action == 'check' and excepts:
        raise Exception('The following files do not contain a valid license, '+
                        'you can use `python tools/license_header.py add` to add'+
                        'them automatically', excepts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add or check source license header')
    parser.add_argument(
        'action', nargs=1, type=str,
        choices=['add', 'check'], default='add',
        help = 'add or check')
    args = parser.parse_args()
    process_folder(os.path.join(os.path.dirname(__file__), '..'), args.action[0])
