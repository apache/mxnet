from __future__ import print_function
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

import logging as log

class Singleton:
    def __init__(self, decorated):
        log.debug("Singleton Init %s" % decorated)
        self._decorated = decorated

    def getInstance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __new__(cls, *args, **kwargs):
        print("__new__")
        cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __call__(self):
        raise TypeError("Singletons must be accessed through 'getInstance()'")

