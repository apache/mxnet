#!/usr/bin/env python
"""
parse mxnet output log into a markdown table
"""
import argparse
import sys
import re

parser = argparse.ArgumentParser(description='Parse mxnet output log')
parser.add_argument('logfile', nargs=1, type=str,
                    help = 'the log file for parsing')
parser.add_argument('--format', type=str, default='markdown',
                    choices = ['markdown', 'none'],
                    help = 'the format of the parsed outout')
args = parser.parse_args()

with open(args.logfile[0]) as f:
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

if args.format == 'markdown':
    print "| epoch | train-accuracy | valid-accuracy | time |"
    print "| --- | --- | --- | --- |"
    for k, v in data.items():
        print "| %2d | %f | %f | %.1f |" % (k+1, v[0]/v[1], v[2]/v[3], v[4]/v[5])
elif args.format == 'none':
    print "epoch\ttrain-accuracy\tvalid-accuracy\ttime"
    for k, v in data.items():
        print "%2d\t%f\t%f\t%.1f" % (k+1, v[0]/v[1], v[2]/v[3], v[4]/v[5])
