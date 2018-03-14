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

import importlib
import sys


def load_cython(package_name, module_name):
    # How to pass something like '.' as package_name?
    name = package_name + '.cyX.' + module_name
    try:
        if sys.version_info >= (3, 0):
            if len(package_name) > 0 and package_name[0] != '.':
                name = package_name + '.cy3.' + module_name
                package_name = None
            else:
                name = 'cy3.' + module_name
        else:
            if len(package_name) > 0 and package_name[0] != '.':
                name = package_name + '.cy2.' + module_name
                package_name = None
            else:
                name = 'cy2.' + module_name
        #print('Attemptiog to load cython module: {}'.format(name))
        the_module = importlib.import_module(name, package=package_name)
        #print('Loaded cython module: {}'.format(name))
        return the_module
    except:
        # No cython found
        print('Unable to load cython module: {}'.format(name))
    return None
