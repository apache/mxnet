#!/usr/bin/env python
#===============================================================================
# Copyright 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

from __future__ import print_function

import os
import re
import sys
import datetime
import xml.etree.ElementTree as ET


def banner(year_from):
    year_now = str(datetime.datetime.now().year)
    banner_year = year_from if year_now == year_from else '%s-%s' % (year_from,
                                                                     year_now)
    return '''\
/*******************************************************************************
* Copyright %s Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

// DO NOT EDIT, AUTO-GENERATED

// clang-format off

''' % banner_year


def template(body, year_from):
    return '%s%s' % (banner(year_from), body)


def header(body):
    return '''\
#ifndef DNNL_DEBUG_H
#define DNNL_DEBUG_H

/// @file
/// Debug capabilities

#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

%s
const char DNNL_API *dnnl_runtime2str(unsigned v);

/// Forms a format string for a given memory descriptor.
///
/// The format is defined as: 'dt:[p|o|0]:fmt_kind:fmt:extra'.
/// Here:
///  - dt       -- data type
///  - p        -- indicates there is non-trivial padding
///  - o        -- indicates there is non-trivial padding offset
///  - 0        -- indicates there is non-trivial offset0
///  - fmt_kind -- format kind (blocked, wino, etc...)
///  - fmt      -- extended format string (format_kind specific)
///  - extra    -- shows extra fields (underspecified)
int DNNL_API dnnl_md2fmt_str(char *fmt_str, size_t fmt_str_len,
        const dnnl_memory_desc_t *md);

/// Forms a dimension string for a given memory descriptor.
///
/// The format is defined as: 'dim0xdim1x...xdimN
int DNNL_API dnnl_md2dim_str(char *dim_str, size_t dim_str_len,
        const dnnl_memory_desc_t *md);

#ifdef __cplusplus
}
#endif

#endif
''' % body


def source(body):
    return '''\
#include <assert.h>

#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_types.h"

%s
''' % body


def header_benchdnn(body):
    return '''\
#ifndef DNNL_DEBUG_HPP
#define DNNL_DEBUG_HPP

#include "oneapi/dnnl/dnnl.h"

%s
/* status */
const char *status2str(dnnl_status_t status);

/* data type */
const char *dt2str(dnnl_data_type_t dt);

/* format */
const char *fmt_tag2str(dnnl_format_tag_t tag);

/* endinge kind */
const char *engine_kind2str(dnnl_engine_kind_t kind);

/* scratchpad mode */
const char *scratchpad_mode2str(dnnl_scratchpad_mode_t mode);

#endif
''' % body


def source_benchdnn(body):
    return '''\
#include <assert.h>
#include <string.h>

#include "oneapi/dnnl/dnnl_debug.h"

#include "dnnl_debug.hpp"

#include "src/common/z_magic.hpp"

%s

const char *status2str(dnnl_status_t status) {
    return dnnl_status2str(status);
}

const char *dt2str(dnnl_data_type_t dt) {
    return dnnl_dt2str(dt);
}

const char *fmt_tag2str(dnnl_format_tag_t tag) {
    return dnnl_fmt_tag2str(tag);
}

const char *engine_kind2str(dnnl_engine_kind_t kind) {
    return dnnl_engine_kind2str(kind);
}

const char *scratchpad_mode2str(dnnl_scratchpad_mode_t mode) {
    return dnnl_scratchpad_mode2str(mode);
}
''' % body.rstrip()


def maybe_skip(enum):
    return enum in (
        'dnnl_memory_extra_flags_t',
        'dnnl_normalization_flags_t',
        'dnnl_query_t',
        'dnnl_rnn_cell_flags_t',
        'dnnl_rnn_packed_memory_format_t',
        'dnnl_stream_flags_t',
        'dnnl_wino_memory_format_t',
        )


def enum_abbrev(enum):
    def_enum = re.sub(r'^dnnl_', '', enum)
    def_enum = re.sub(r'_t$', '', def_enum)
    return {
        'dnnl_data_type_t': 'dt',
        'dnnl_format_kind_t': 'fmt_kind',
        'dnnl_format_tag_t': 'fmt_tag',
        'dnnl_primitive_kind_t': 'prim_kind',
        'dnnl_engine_kind_t': 'engine_kind',
    }.get(enum, def_enum)


def sanitize_value(v):
    if 'undef' in v:
        return 'undef'
    if 'any' in v:
        return 'any'
    v = v.split('dnnl_scratchpad_mode_')[-1]
    v = v.split('dnnl_format_kind_')[-1]
    v = v.split('dnnl_')[-1]
    return v


def func_to_str_decl(enum, is_header=False):
    abbrev = enum_abbrev(enum)
    return 'const char %s*dnnl_%s2str(%s v)' % \
        ('DNNL_API ' if is_header else '', abbrev, enum)


def func_to_str(enum, values):
    indent = '    '
    abbrev = enum_abbrev(enum)
    func = ''
    func += func_to_str_decl(enum) + ' {\n'
    for v in values:
        func += '%sif (v == %s) return "%s";\n' \
                % (indent, v, sanitize_value(v))
    func += '%sassert(!"unknown %s");\n' % (indent, abbrev)
    func += '%sreturn "unknown %s";\n}\n' % (indent, abbrev)
    return func


def str_to_func_decl(enum, is_header=False, is_dnnl=True):
    attr = 'DNNL_API ' if is_header and is_dnnl else ''
    prefix = 'dnnl_' if is_dnnl else ''
    abbrev = enum_abbrev(enum)
    return '%s %s%sstr2%s(const char *str)' % \
        (enum, attr, prefix, abbrev)


def str_to_func(enum, values, is_dnnl=True):
    indent = '    '
    abbrev = enum_abbrev(enum)
    func = ''
    func += str_to_func_decl(enum, is_dnnl=is_dnnl) + ' {\n'
    func += '''#define CASE(_case) do { \\
    if (!strcmp(STRINGIFY(_case), str) \\
            || !strcmp("dnnl_" STRINGIFY(_case), str)) \\
        return CONCAT2(dnnl_, _case); \\
} while (0)
'''
    special_values = []
    for v in values:
        if 'last' in v:
            continue
        if 'undef' in v:
            v_undef = v
            special_values.append(v)
            continue
        if 'any' in v:
            special_values.append(v)
            continue
        func += '%sCASE(%s);\n' % (indent, sanitize_value(v))
    func += '#undef CASE\n'
    for v in special_values:
        v_short = re.search(r'(any|undef)', v).group()
        func += '''%sif (!strcmp("%s", str) || !strcmp("%s", str))
        return %s;
''' % (indent, v_short, v, v)
    if enum != 'dnnl_format_tag_t':
        func += '%sassert(!"unknown %s");\n' % (indent, abbrev)
    func += '%sreturn %s;\n}\n' % (indent,
                                   v_undef if enum != 'dnnl_format_tag_t' else
                                   'dnnl_format_tag_last')
    return func


def generate(ifile, banner_years):
    h_body, s_body = '', ''
    h_benchdnn_body, s_benchdnn_body = '', ''
    root = ET.parse(ifile).getroot()
    for v_enum in root.findall('Enumeration'):
        enum = v_enum.attrib['name']
        if maybe_skip(enum):
            continue
        values = [v_value.attrib['name']
                  for v_value in v_enum.findall('EnumValue')]
        h_body += func_to_str_decl(enum, is_header=True) + ';\n'
        s_body += func_to_str(enum, values) + '\n'
        if enum in ['dnnl_format_tag_t', 'dnnl_data_type_t']:
            h_benchdnn_body += str_to_func_decl(
                enum, is_header=True, is_dnnl=False) + ';\n'
            s_benchdnn_body += str_to_func(
                enum, values, is_dnnl=False) + '\n'
    bodies = [
        header(h_body),
        source(s_body),
        header_benchdnn(h_benchdnn_body),
        source_benchdnn(s_benchdnn_body)
    ]
    return [template(b, y) for b, y in zip(bodies, banner_years)]


def usage():
    print('''\
%s types.xml

Generates oneDNN debug header and source files with enum to string mapping.
Input types.xml file can be obtained with CastXML[1]:
$ castxml --castxml-cc-gnu-c clang --castxml-output=1 \\
        include/oneapi/dnnl/dnnl_types.h -o types.xml

[1] https://github.com/CastXML/CastXML''' % sys.argv[0])
    sys.exit(1)


for arg in sys.argv:
    if '-help' in arg:
        usage()

script_root = os.path.dirname(os.path.realpath(__file__))

ifile = sys.argv[1] if len(sys.argv) > 1 else usage()

file_paths = (
    '%s/../include/oneapi/dnnl/dnnl_debug.h' % script_root,
    '%s/../src/common/dnnl_debug_autogenerated.cpp' % script_root,
    '%s/../tests/benchdnn/dnnl_debug.hpp' % script_root,
    '%s/../tests/benchdnn/dnnl_debug_autogenerated.cpp' % script_root)

banner_years = []
for file_path in file_paths:
    with open(file_path, 'r') as f:
        m = re.search(r'Copyright (.*) Intel', f.read())
        banner_years.append(m.group(1).split('-')[0])

for file_path, file_body in zip(file_paths, generate(ifile, banner_years)):
    with open(file_path, 'w') as f:
        f.write(file_body)
