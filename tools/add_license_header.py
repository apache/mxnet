import re
import os

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

_LANGS = {'.cc':'*', '.h':'*', '.cu':'*', '.cuh':'*', '.py':'#',
          '.pm':'#', '.scala':'*', '.cc':'*', '.sh':'#'}
_OLD_LICENSE = re.compile('.*Copyright.*by Contributors')

def _get_license(comment_mark):
    body = ['%s %s'%(comment_mark, l) for l in _LICENSE.split('\n')]
    if comment_mark == '*':
        return '/*\n' + '\n'.join([' '+l for l in body])+'\n */\n\n'
    else:
        return '\n'.join(body) + '\n\n'

def process_file(fname):
    _, ext = os.path.splitext(fname)
    if ext not in _LANGS:
        return
    with open(fname, 'r') as f:
        lines = f.readlines()
    if not len(lines):
        return
    has_license_header = False
    for l in lines:
        if 'Licensed to the Apache Software Foundation' in l:
            has_license_header = True
            break
    if has_license_header:
        return
    # remove old license
    if ext == '.h' or ext == '.cc' or ext == '.cu':
        for i, l in enumerate(lines):
            if _OLD_LICENSE.match(l):
                del lines[i]
                break
    with open(fname, 'w') as f:
        # shebang line
        if lines[0].startswith('#!'):
            f.write(lines[0]+'\n')
            del lines[0]
        f.write(_get_license(_LANGS[ext]))
        for l in lines:
            f.write(l)
    print('added license header to ' + fname)

def process_folder(root):
    for root, _, files in os.walk(root):
        for f in files:
            process_file(os.path.join(root, f))
if __name__ == '__main__':
    # process_folder('../src/operator/tensor')
    process_file('../python/mxnet/registry.py')
