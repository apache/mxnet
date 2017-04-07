#!/usr/bin/env python
"""Manipulate codes blocks in a markdown file"""
import argparse
import re
import os
import codecs
import json

# language names and the according file extensions
_LANGS = {'python', 'r', 'scala', 'julia', 'perl', 'cpp'}

# start or end of a code block
_CODE_MARK = re.compile('^([ ]*)```([\w]*)')

class CodeBlocks(object):
    def __init__(self, fname, lang):
        with codecs.open(fname, 'r', 'utf-8') as f:
            self.data = f.readlines()
        self.lang = lang.lower()
        self.cells = []

    def _parse_lines(self):
        in_code = False
        lang = None
        indent = None
        for l in self.data:
            m = _CODE_MARK.match(l)
            if m is not None:
                if not in_code:
                    if m.groups()[1].lower() in _LANGS:
                        lang = m.groups()[1].lower()
                        indent = len(m.groups()[0])
                        in_code = True
                    yield (l, in_code, lang, indent)
                else:
                    yield (l, in_code, lang, indent)
                    lang = None
                    indent = None
                    in_code = False
            else:
                yield (l, in_code, lang, indent)

    def _add_jupyter_block(self, lines, is_code ):
        if is_code and len(lines) >= 2:
            lines = lines[1:-1] # remove ```
        while len(lines) > 0:
            if len(lines[0].rstrip()) == 0:
                lines.pop(0)
            else:
                break
        while len(lines) > 0:
            if len(lines[-1].rstrip()) == 0:
                lines.pop()
            else:
                break
        if len(lines) == 0:
            return
        lines[-1] = lines[-1].rstrip()
        cell = {
            "cell_type": "code" if is_code else "markdown",
            "metadata": {},
            "source":  lines
        }
        if is_code:
            cell.update({
                "outputs": [],
                "execution_count": None,
            })
        self.cells.append(cell)

    def write(self, action, ofname):
        if action == 'get':
            with open(ofname, 'w') as f:
                for (l, in_code, lang, indent) in self._parse_lines():
                    if in_code and lang == self.lang and l[indent:indent+3] != '```':
                        f.write(l[indent:])
            return
        if action == 'keep':
            with open(ofname, 'w') as f:
                for (l, in_code, lang, _) in self._parse_lines():
                    if not in_code or in_code and lang == self.lang:
                        f.write(l)
            return
        if action == 'convert':
            cur_block = []
            pre_in_code = None
            pre_lang = None
            for (l, in_code, lang, _) in self._parse_lines():
                if in_code != pre_in_code or lang != pre_lang:
                    self._add_jupyter_block(cur_block, pre_in_code)
                    cur_block = []
                if not in_code or (in_code and lang == self.lang):
                    cur_block.append(l)
                (pre_in_code, pre_lang) = (in_code, lang)
            self._add_jupyter_block(cur_block, pre_in_code)

            ipynb = {"nbformat":4, "nbformat_minor":2,
                     "metadata":{"language":self.lang, "display_name":'', "name":''}, "cells":self.cells}
            with open(ofname, 'w') as f:
                json.dump(ipynb, f)
            return
        if action == 'add_btn':
            langs = set([l for (_, _, l, _) in self._parse_lines() if l is not None])
            print langs
            active = True
            btngroup = """<div class="text-center">
<div class="btn-group opt-group" role="group">
"""
            for l in langs:
                btngroup += "<button type=\"button\" class=\"btn btn-default opt %s\">%s</button>\n" % (
                    'active' if active else '', l[0].upper()+l[1:].lower())
                active = False
            btngroup += """</div>
</div>
<script type="text/javascript" src='../../_static/js/options.js'></script>
"""
            with open(ofname, 'w') as f:
                for l in self.data:
                    if 'ENABLE LANGUAGE BAR' in l:
                        f.write(btngroup)
                    else:
                        f.write(l)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Manipulate code blocks in markdown files.
Sample usage:
- extract all python code blocks in example
  ./mdcode.py get python example.md example.py
- remove all codes blocks except for python
  ./mdcode.py keep python example.md example_py.md
- remove all codes blocks except for python and then convert into jupyter notebook
  ./mdcode.py convert python example.md example.ipynb
- add the language selection botton group into example.md
  ./mdcode.py add_btn all example.md example.md
    """)
    parser.add_argument('action', help='action',
                        choices=['get', 'keep', 'convert', 'add_btn'])
    parser.add_argument('lang', help='code language')
    parser.add_argument('input', help='input markdown filename')
    parser.add_argument('output', help='output file')
    args = parser.parse_args()

    code_blocks = CodeBlocks(args.input, args.lang)
    code_blocks.write(args.action, args.output)
