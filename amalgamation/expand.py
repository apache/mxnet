import os.path, re, StringIO

blacklist = [
'Windows.h', 'cublas_v2.h', 'cuda/tensor_gpu-inl.cuh', 'cuda_runtime.h', 'cudnn.h', 'cudnn_lrn-inl.h', 'curand.h', 'glog/logging.h', 'io/azure_filesys.h', 'io/hdfs_filesys.h', 'io/s3_filesys.h', 'kvstore_dist.h', 'mach/clock.h', 'mach/mach.h', 'malloc.h', 'mkl.h', 'mkl_cblas.h', 'mkl_vsl.h', 'mkl_vsl_functions.h', 'nvml.h', 'opencv2/opencv.hpp', 'sys/stat.h', 'sys/types.h', 'emmintrin.h'
]

sources = []
files = []
# g++ -MD -MF mxnet0.d -std=c++11 -Wall -I./mshadow/ -I./dmlc-core/include -Iinclude  -I/usr/local//Cellar/openblas/0.2.14_1/include -c  mxnet0.cc
for line in open('mxnet0.d'):
	files = files + line.strip().split(' ')

for f in files:
	f = f.strip()
	if not f or f == 'mxnet0.o:' or f == '\\': continue
	fn = os.path.relpath(f)
	if fn.find('/usr/') < 0 and fn not in sources:
		sources.append(fn)

def find_source(name, start):
	candidates = []
	for x in sources:
		if x == name or x.endswith('/' + name): candidates.append(x)
	if not candidates: return ''
	if len(candidates) == 1: return candidates[0]
	for x in candidates:
#		print 'multiple candidates: %s, looking for %s, candidates: %s' %(start, name, str(candidates))
		if x.split('/')[1] == start.split('/')[1]: return x
	return ''


re1 = re.compile('<([./a-zA-Z0-9_-]*)>')
re2 = re.compile('"([./a-zA-Z0-9_-]*)"')

sysheaders = []
history = {}

out = StringIO.StringIO()
def expand(x, pending):
	if x in history and x not in ['mshadow/mshadow/expr_scalar-inl.h']: # MULTIPLE includes
		return
	
	if x in pending:
#		print 'loop found: %s in ' % x, pending
		return

	print >>out, "//===== EXPANDIND: %s =====\n" %x
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
	history[x] = 1

expand('mxnet0.cc', [])

f = open('mxnet.cc', 'wb')
print >>f, '''
#if defined(__MACH__)
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#if !defined(__WIN32__)
#include <sys/stat.h>
#include <sys/types.h>

#if !defined(__ANDROID__)
#include <emmintrin.h>
#endif

#endif
'''

for k in sorted(sysheaders):
	print >>f, "#include <%s>" % k

print >>f, ''
print >>f, out.getvalue()

for x in sources:
	if x not in history: print 'Not processed:', x

