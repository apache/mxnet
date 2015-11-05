#!/usr/bin/env python
import sys
import re

if len(sys.argv) != 2:
    print "parse mxnet output log into a markdown table"
    print "usage: %s log_file" % (sys.argv[0])
    exit(-1)

with open(sys.argv[1]) as f:
    lines = f.readlines()

res = [re.compile('.*Epoch\[(\d+)\] Train.*=([.\d]+)'),
       re.compile('.*Epoch\[(\d+)\] Valid.*=([.\d]+)'),
       re.compile('.*Epoch\[(\d+)\] Time.*=([.\d]+)')]

data = {}
for l in lines:
    i = 0
    for r in res:
        m = r.match(l)
        if m is not None:
            break
        i += 1
    if m is None:
        continue

    assert len(m.groups()) == 2
    epoch = int(m.groups()[0])
    val = float(m.groups()[1])

    if epoch not in data:
        data[epoch] = [0] * len(res) * 2

    data[epoch][i*2] += val
    data[epoch][i*2+1] += 1

print "| epoch | train accuracy | valid accuracy | time |"
print "| --- | --- | --- | --- |"
for k, v in data.items():
    print "| %d | %f | %f | %.1f |" % (k+1, v[0]/v[1], v[2]/v[3], v[4]/v[5])
