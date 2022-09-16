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

import sys
import os
import re

def has_token(token, lines):
    for line in lines:
        if token in line:
            return True
    return False

def get_next_title_mark(lines):
    available_marks = ['=', '-', '~', '^']
    for mark in available_marks:
        if has_token(mark*3, lines):
            continue
        else:
            return mark
    return None

def add_hidden_title(inputs):
    """
    convert

       .. autoclass:: Class

    into

       :hidden:`Class`
       ~~~~~~~~~~~~~~~

       .. autoclass:: Class
    """
    lines = inputs.split('\n')
    if not has_token('doxygenfunction:', lines):
        return inputs, None

    outputs = """.. raw:: html

   <div class="mx-api">

.. role:: hidden
    :class: hidden-section

"""
    num = 0
    FUNC = re.compile('\.\. doxygenfunction\:\:[ ]+([\w\.]+)')
    mark = get_next_title_mark(lines)
    assert mark is not None
    for line in lines:
        m = FUNC.match(line)
        if m is not None:
            name = ':hidden:`' + m.groups()[0] + '`'
            outputs += '\n' + name + '\n' + mark * len(name) + '\n\n'
            num += 1
        outputs += line + '\n'
    outputs += '.. raw:: html\n\n    </div>\n'
    return outputs, num


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'usage: input.rst output.rst'
    (src_fn, input_fn, output_fn) = sys.argv
    with open(input_fn, 'r') as f:
        inputs = f.read()
    outputs, num = add_hidden_title(inputs)
    if num is not None:
        print(f'{src_fn}: add {num} hidden sections for {input_fn}')
    with open(output_fn, 'w') as f:
        f.write(outputs)
