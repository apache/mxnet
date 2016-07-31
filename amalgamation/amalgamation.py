import sys
import os.path, re, StringIO

blacklist = [
    'Windows.h', 'cublas_v2.h', 'cuda/tensor_gpu-inl.cuh',
    'cuda_runtime.h', 'cudnn.h', 'cudnn_lrn-inl.h', 'curand.h',
    'glog/logging.h', 'io/azure_filesys.h', 'io/hdfs_filesys.h', 'io/s3_filesys.h',
    'kvstore_dist.h', 'mach/clock.h', 'mach/mach.h',
    'malloc.h', 'mkl.h', 'mkl_cblas.h', 'mkl_vsl.h', 'mkl_vsl_functions.h',
    'nvml.h', 'opencv2/opencv.hpp', 'sys/stat.h', 'sys/types.h', 'cuda.h', 'cuda_fp16.h'
    ]

if len(sys.argv) < 4:
    print("Usage: <source.d> <source.cc> <output> [minimum=0] [android=0]\n"
          "Minimum means no blas, no sse, no dependency, may run twice slower.")
    exit(0)

minimum = int(sys.argv[4]) if len(sys.argv) > 4 else 0
android = int(sys.argv[5]) if len(sys.argv) > 5 else 0

if minimum:
    blacklist += ['packet/sse-inl.h', 'emmintrin.h', 'cblas.h']

if android:
    blacklist += ['config.h']
    if 'packet/sse-inl.h' not in blacklist:
        blacklist += ['packet/sse-inl.h']

def get_sources(def_file):
    sources = []
    files = []
    visited = set()
    mxnet_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    for line in open(def_file):
        files = files + line.strip().split(' ')

    for f in files:
        f = f.strip()
        if not f or f.endswith('.o:') or f == '\\': continue
        fn = os.path.relpath(f)
        if os.path.abspath(f).startswith(mxnet_path) and fn not in visited:
            sources.append(fn)
            visited.add(fn)
    return sources

sources = get_sources(sys.argv[1])

def find_source(name, start):
    candidates = []
    for x in sources:
        if x == name or x.endswith('/' + name): candidates.append(x)
    if not candidates: return ''
    if len(candidates) == 1: return candidates[0]
    for x in candidates:
        if x.split('/')[1] == start.split('/')[1]: return x
    return ''


re1 = re.compile('<([./a-zA-Z0-9_-]*)>')
re2 = re.compile('"([./a-zA-Z0-9_-]*)"')

sysheaders = []
history = set([])
out = StringIO.StringIO()

def expand(x, pending):
    if x in history and x not in ['mshadow/mshadow/expr_scalar-inl.h']: # MULTIPLE includes
        return

    if x in pending:
        #print 'loop found: %s in ' % x, pending
        return

    print >>out, "//===== EXPANDING: %s =====\n" %x
    for line in open(x):
        if line.find('#include') < 0:
            out.write(line)
            continue
        if line.strip().find('#include') > 0:
            print line
            continue
        m = re1.search(line)
        if not m: m = re2.search(line)
        if not m:
            print line + ' not found'
            continue
        h = m.groups()[0].strip('./')
        source = find_source(h, x)
        if not source:
            if h not in blacklist and h not in sysheaders: sysheaders.append(h)
        else:
            expand(source, pending + [x])
    print >>out, "//===== EXPANDED: %s =====\n" %x
    history.add(x)


expand(sys.argv[2], [])

f = open(sys.argv[3], 'wb')

if minimum != 0:
    print >>f, "#define MSHADOW_STAND_ALONE 1"
    print >>f, "#define MSHADOW_USE_SSE 0"
    print >>f, "#define MSHADOW_USE_CBLAS 0"

print >>f, '''
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
'''

if minimum != 0 and android != 0 and 'complex.h' not in sysheaders:
    sysheaders.append('complex.h')

for k in sorted(sysheaders):
    print >>f, "#include <%s>" % k

print >>f, ''
print >>f, out.getvalue()

for x in sources:
    if x not in history and not x.endswith('.o'):
        print 'Not processed:', x

