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
import inspect

def load_module(package_name, module_name):
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
        the_module = importlib.import_module(name, package=package_name)
        return the_module
    except:
        # No cython found
        print('Unable to load cython module: {}'.format(name))
    return None


def module_name(skip=1, prune=0):
    """Get a name of a caller in the format module.class.method

       `skip` specifies how many levels of stack to skip while getting caller
       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

       An empty string is returned if skipped levels exceed stack height
    """
    stack = inspect.stack()
    start = 0 + skip
    if len(stack) < start + 1:
        return ''
    parentframe = stack[start][0]

    name = []
    module = inspect.getmodule(parentframe)
    # `modname` can be None when frame is executed directly in console
    # TODO(techtonik): consider using __main__
    if module:
        name.append(module.__name__)
    # detect classname
    if 'self' in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # XXX: there seems to be no way to detect static method call - it will
        #      be just a function call
        name.append(parentframe.f_locals['self'].__class__.__name__)
    codename = parentframe.f_code.co_name
    if codename != '<module>':  # top level usually
        name.append(codename)  # function or a method
    del parentframe
    res = ".".join(name)
    if prune > 0:
      items = res.split('.')
      res = ''
      count = len(items) - prune
      for i in range(count):
        if i > 0:
          res += '.'
        res += items[i]
    return res


# Global to control the use of cython at runtime
_use_cython = True

