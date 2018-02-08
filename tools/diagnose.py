#!/usr/bin/env python

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

"""Diagnose script for checking OS/hardware/python/pip/mxnet/network.
The output of this script can be a very good hint to issue/problem.
"""
import platform, subprocess, sys, os
import socket, time
try:
    from urllib.request import urlopen
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
    from urllib2 import urlopen
import argparse

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Diagnose script for checking the current system.')
    choices = ['python', 'pip', 'mxnet', 'os', 'hardware', 'network']
    for choice in choices:
        parser.add_argument('--' + choice, default=1, type=int,
                            help='Diagnose {}.'.format(choice))
    parser.add_argument('--region', default='', type=str,
                        help="Additional sites in which region(s) to test. \
                        Specify 'cn' for example to test mirror sites in China.")
    parser.add_argument('--timeout', default=10, type=int,
                        help="Connection test timeout threshold, 0 to disable.")
    args = parser.parse_args()
    return args

URLS = {
    'MXNet': 'https://github.com/apache/incubator-mxnet',
    'Gluon Tutorial(en)': 'http://gluon.mxnet.io',
    'Gluon Tutorial(cn)': 'https://zh.gluon.ai',
    'FashionMNIST': 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/fashion-mnist/train-labels-idx1-ubyte.gz',
    'PYPI': 'https://pypi.python.org/pypi/pip',
    'Conda': 'https://repo.continuum.io/pkgs/free/',
}
REGIONAL_URLS = {
    'cn': {
        'PYPI(douban)': 'https://pypi.douban.com/',
        'Conda(tsinghua)': 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/',
    }
}

def test_connection(name, url, timeout=10):
    """Simple connection test"""
    urlinfo = urlparse(url)
    start = time.time()
    try:
        ip = socket.gethostbyname(urlinfo.netloc)
    except Exception as e:
        print('Error resolving DNS for {}: {}, {}'.format(name, url, e))
        return
    dns_elapsed = time.time() - start
    start = time.time()
    try:
        _ = urlopen(url, timeout=timeout)
    except Exception as e:
        print("Error open {}: {}, {}, DNS finished in {} sec.".format(name, url, e, dns_elapsed))
        return
    load_elapsed = time.time() - start
    print("Timing for {}: {}, DNS: {:.4f} sec, LOAD: {:.4f} sec.".format(name, url, dns_elapsed, load_elapsed))

def check_python():
    print('----------Python Info----------')
    print('Version      :', platform.python_version())
    print('Compiler     :', platform.python_compiler())
    print('Build        :', platform.python_build())
    print('Arch         :', platform.architecture())

def check_pip():
    print('------------Pip Info-----------')
    try:
        import pip
        print('Version      :', pip.__version__)
        print('Directory    :', os.path.dirname(pip.__file__))
    except ImportError:
        print('No corresponding pip install for current python.')

def check_mxnet():
    print('----------MXNet Info-----------')
    try:
        import mxnet
        print('Version      :', mxnet.__version__)
        mx_dir = os.path.dirname(mxnet.__file__)
        print('Directory    :', mx_dir)
        commit_hash = os.path.join(mx_dir, 'COMMIT_HASH')
        with open(commit_hash, 'r') as f:
            ch = f.read().strip()
            print('Commit Hash   :', ch)
    except ImportError:
        print('No MXNet installed.')
    except IOError:
        print('Hashtag not found. Not installed from pre-built package.')
    except Exception as e:
        import traceback
        if not isinstance(e, IOError):
            print("An error occured trying to import mxnet.")
            print("This is very likely due to missing missing or incompatible library files.")
        print(traceback.format_exc())

def check_os():
    print('----------System Info----------')
    print('Platform     :', platform.platform())
    print('system       :', platform.system())
    print('node         :', platform.node())
    print('release      :', platform.release())
    print('version      :', platform.version())

def check_hardware():
    print('----------Hardware Info----------')
    print('machine      :', platform.machine())
    print('processor    :', platform.processor())
    if sys.platform.startswith('darwin'):
        pipe = subprocess.Popen(('sysctl', '-a'), stdout=subprocess.PIPE)
        output = pipe.communicate()[0]
        for line in output.split(b'\n'):
            if b'brand_string' in line or b'features' in line:
                print(line.strip())
    elif sys.platform.startswith('linux'):
        subprocess.call(['lscpu'])
    elif sys.platform.startswith('win32'):
        subprocess.call(['wmic', 'cpu', 'get', 'name'])

def check_network(args):
    print('----------Network Test----------')
    if args.timeout > 0:
        print('Setting timeout: {}'.format(args.timeout))
        socket.setdefaulttimeout(10)
    for region in args.region.strip().split(','):
        r = region.strip().lower()
        if not r:
            continue
        if r in REGIONAL_URLS:
            URLS.update(REGIONAL_URLS[r])
        else:
            import warnings
            warnings.warn('Region {} do not need specific test, please refer to global sites.'.format(r))
    for name, url in URLS.items():
        test_connection(name, url, args.timeout)

if __name__ == '__main__':
    args = parse_args()
    if args.python:
        check_python()

    if args.pip:
        check_pip()

    if args.mxnet:
        check_mxnet()

    if args.os:
        check_os()

    if args.hardware:
        check_hardware()

    if args.network:
        check_network(args)
