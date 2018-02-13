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

from __future__ import print_function

import sys
import os.path, re
from io import BytesIO, StringIO
import platform

blacklist = [
    'Windows.h', 'cublas_v2.h', 'cuda/tensor_gpu-inl.cuh',
    'cuda_runtime.h', 'cudnn.h', 'cudnn_lrn-inl.h', 'curand.h', 'curand_kernel.h',
    'glog/logging.h', 'io/azure_filesys.h', 'io/hdfs_filesys.h', 'io/s3_filesys.h',
    'kvstore_dist.h', 'mach/clock.h', 'mach/mach.h',
    'malloc.h', 'mkl.h', 'mkl_cblas.h', 'mkl_vsl.h', 'mkl_vsl_functions.h',
    'nvml.h', 'opencv2/opencv.hpp', 'sys/stat.h', 'sys/types.h', 'cuda.h', 'cuda_fp16.h',
    'omp.h', 'execinfo.h', 'packet/sse-inl.h', 'emmintrin.h', 'thrust/device_vector.h',
    'cusolverDn.h', 'internal/concurrentqueue_internal_debug.h', 'relacy/relacy_std.hpp',
    'relacy_shims.h'
    ]

minimum = int(sys.argv[6]) if len(sys.argv) > 5 else 0
android = int(sys.argv[7]) if len(sys.argv) > 6 else 0

# blacklist linear algebra headers when building without blas.
if minimum != 0:
    blacklist.append('linalg.h')

if platform.system() != 'Darwin':
    blacklist.append('TargetConditionals.h')

if platform.system() != 'Windows':
    blacklist.append('windows.h')
    blacklist.append('process.h')


def get_sources(def_file):
    sources = []
    files = []
    visited = set()
    mxnet_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    for line in open(def_file):
        files = files + line.strip().split(' ')

    for f in files:
        f = f.strip()
        if not f or f.endswith('.o:') or f == '\\': continue
        f = os.path.realpath(f)
        fn = os.path.relpath(f)
        if f.startswith(mxnet_path) and fn not in visited:
            sources.append(fn)
            visited.add(fn)
    return sources


sources = get_sources(sys.argv[1])


def find_source(name, start, stage):
    candidates = []
    for x in sources:
        if x == name:
            candidates.append(x)
        elif name.endswith(".cc") and x.endswith('/' + name):
            if x.startswith("../" + stage):
                candidates.append(x)
        elif x.endswith('/' + name):
            candidates.append(x)
        #if x == name or x.endswith('/' + name): candidates.append(x)
    if not candidates: return ''
    if len(candidates) == 1: return candidates[0]
    for x in candidates:
        if x.split('/')[1] == start.split('/')[1]: return x
    return ''


re1 = re.compile('<([./a-zA-Z0-9_-]*)>')
re2 = re.compile('"([./a-zA-Z0-9_-]*)"')

sysheaders = []
history = set([])
out = BytesIO()


def expand(x, pending, stage):
    if x in history and x not in ['mshadow/mshadow/expr_scalar-inl.h']: # MULTIPLE includes
        return

    if x in pending:
        #print('loop found: {} in {}'.format(x, pending))
        return

    whtspace = '  ' * expand.treeDepth
    expand.fileCount += 1
    comment = u"//=====[{:3d}] STAGE:{:>4} {}EXPANDING: {} =====\n\n".format(expand.fileCount, stage, whtspace, x)
    out.write(comment.encode('ascii'))
    print(comment)

    with open(x, 'rb') as x_h:
        for line in x_h.readlines():
            uline = line.decode('utf-8')
            if uline.find('#include') < 0:
                out.write(line)
                continue
            if uline.strip().find('#include') > 0:
                print(uline)
                continue
            m = re1.search(uline)
            if not m:
                m = re2.search(uline)
            if not m:
                print(uline + ' not found')
                continue
            h = m.groups()[0].strip('./')
            source = find_source(h, x, stage)
            if not source:
                if (h not in blacklist and
                    h not in sysheaders and
                    'mkl' not in h and
                    'nnpack' not in h and
                    not h.endswith('.cuh')): sysheaders.append(h)
            else:
                expand.treeDepth += 1
                expand(source, pending + [x], stage)
                expand.treeDepth -= 1

    out.write(u"//===== EXPANDED  : {} =====\n\n".format(x).encode('ascii'))
    history.add(x)


# Vars to keep track of number of files expanded.
# Used in printing informative comments.
expand.treeDepth = 0
expand.fileCount = 0

# Expand the stages
expand(sys.argv[2], [], "dmlc")
expand(sys.argv[3], [], "nnvm")
expand(sys.argv[4], [], "src")

# Write to amalgamation file
with open(sys.argv[5], 'wb') as f:

    if minimum != 0:
        sysheaders.remove('cblas.h')
        f.write(b"#define MSHADOW_STAND_ALONE 1\n")
        f.write(b"#define MSHADOW_USE_SSE 0\n")
        f.write(b"#define MSHADOW_USE_CBLAS 0\n")

    f.write(
        b"""
#if defined(__MACH__)
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#if !defined(__WIN32__)
#include <sys/stat.h>
#include <sys/types.h>

#if !defined(__ANDROID__) && (!defined(MSHADOW_USE_SSE) || MSHADOW_USE_SSE == 1)
#include <emmintrin.h>
#endif

#endif
\n"""
    )

    if minimum != 0 and android != 0 and 'complex.h' not in sysheaders:
        sysheaders.append('complex.h')

    for k in sorted(sysheaders):
        f.write("#include <{}>\n".format(k).encode('ascii'))

    f.write(b'\n')
    f.write(out.getvalue())
    f.write(b'\n')

for src in sources:
    if src not in history and not src.endswith('.o'):
        print('Not processed:', src)


