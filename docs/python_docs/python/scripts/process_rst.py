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
        print('%s: add %d hidden sections for %s' % (src_fn, num, input_fn))
    with open(output_fn, 'w') as f:
        f.write(outputs)
