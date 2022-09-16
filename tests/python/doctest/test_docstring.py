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

import doctest
import logging
import mxnet
import numpy

def import_into(globs, module, names=None, error_on_overwrite=True):
    """Import names from module into the globs dict.

    Parameters
    ----------
    """
    mod_names = dir(module)
    if names is not None:
        for name in names:
            assert name in mod_names, f'{name} not found in {module}'
        mod_names = names

    for name in mod_names:
        if name in globs and globs[name] is not getattr(module, name):
            error_msg = f'Attempting to overwrite definition of {name}'
            if error_on_overwrite:
                raise RuntimeError(error_msg)
            logging.warning('%s', error_msg)
        globs[name] = getattr(module, name)

    return globs


def test_symbols():
    globs = {'np': numpy, 'mx': mxnet, 'test_utils': mxnet.test_utils, 'SymbolDoc': mxnet.symbol_doc.SymbolDoc}

    # make sure all the operators are available
    import_into(globs, mxnet.symbol)
    doctest.testmod(mxnet.symbol_doc, globs=globs, verbose=True)

def test_ndarray():
    globs = {'np': numpy, 'mx': mxnet}

    doctest.testmod(mxnet.ndarray, globs=globs, verbose=True)


if __name__ == '__main__':
    test_symbols()
    test_ndarray()
