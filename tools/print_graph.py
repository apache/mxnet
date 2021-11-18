#!/usr/bin/env python3

import re
import sys


RE_NODE = re.compile(r'node\s(.+)\n')
RE_ATTR = re.compile(r'attr\s(.+)\n')
RE_INP = re.compile(r'inp\s(.+)\n')
RE_DEP = re.compile(r'dep\s(.+)\n')
RE_SUB = re.compile(r'node\s(.+)\n')


def to_dot(f):
    print('digraph Net {')
    for line in f:
        m = RE_NODE.fullmatch(line)
        if m:
            nid, name, op = m.group(1).split()
            shape = 'ellipse' if op == '(var)' else 'rectangle'
            print(f'  node_{nid} [shape={shape}, label={name}]')
            continue
        m = RE_ATTR.fullmatch(line)
        if m:
            continue
        m = RE_INP.fullmatch(line)
        if m:
            njd, _name, index, _version = m.group(1).split()
            print(f'  node_{njd} -> node_{nid} [label={index}, style=solid]')
            continue
        m = RE_DEP.fullmatch(line)
        if m:
            njd, _name = m.group(1).split()
            print(f'  node_{njd} -> node_{nid} [style=dashed]')
            continue
        m = RE_SUB.fullmatch(line)
        if m:
            continue
        break
    print('}')


if __name__ == '__main__':
    to_dot(sys.stdin)
