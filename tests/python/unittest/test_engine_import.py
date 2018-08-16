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

import os

try:
    reload         # Python 2
except NameError:  # Python 3
    from importlib import reload


def test_engine_import():
    import mxnet
        
    engine_types = ['', 'NaiveEngine', 'ThreadedEngine', 'ThreadedEnginePerDevice']

    for type in engine_types:
        if type:
            os.environ['MXNET_ENGINE_TYPE'] = type
        else:
            os.environ.pop('MXNET_ENGINE_TYPE', None)
        reload(mxnet)


if __name__ == '__main__':
    import nose
    nose.runmodule()
