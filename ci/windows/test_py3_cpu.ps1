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

7z x -y windows_package.7z

$env:MXNET_LIBRARY_PATH=join-path $pwd.Path windows_package\lib\libmxnet.dll
$env:PYTHONPATH=join-path $pwd.Path windows_package\python
$env:MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
$env:MXNET_SUBGRAPH_VERBOSE=0
$env:MXNET_HOME=[io.path]::combine($PSScriptRoot, 'mxnet_home')

C:\Python37\Scripts\pip install -r ci\docker\install\requirements
C:\Python37\python.exe -m pytest -v -m 'not serial' -n 4 --durations=50 --cov-report xml:tests_unittest.xml tests\python\unittest
if ($LastExitCode -ne 0) { Throw ("Error running parallel unittest, python exited with status code " + ('{0:X}' -f $LastExitCode)) }
C:\Python37\python.exe -m pytest -v -m 'serial' --durations=50 --cov-report xml:tests_unittest.xml --cov-append tests\python\unittest
if ($LastExitCode -ne 0) { Throw ("Error running serial unittest, python exited with status code " + ('{0:X}' -f $LastExitCode)) }
C:\Python37\python.exe -m pytest -v -m 'not serial' -n 4 --durations=50 --cov-report xml:tests_train.xml tests\python\train
if ($LastExitCode -ne 0) { Throw ("Error running parallel train tests, python exited with status code " + ('{0:X}' -f $LastExitCode)) }
C:\Python37\python.exe -m pytest -v -m 'serial' --durations=50 --cov-report xml:tests_train.xml --cov-append tests\python\train
if ($LastExitCode -ne 0) { Throw ("Error running serial train tests, python exited with status code " + ('{0:X}' -f $LastExitCode)) }
# Adding this extra test since it's not possible to set env var on the fly in Windows.
C:\Python37\python.exe -m pytest -v --durations=50 --cov-report xml:tests_unittest.xml --cov-append tests\python\unittest\test_operator.py::test_norm
if ($LastExitCode -ne 0) { Throw ("Error running unittest, python exited with status code " + ('{0:X}' -f $LastExitCode)) }

# Need to explicitly set the environment variable for MXNET_MEMORY_OPT.
$env:MXNET_MEMORY_OPT=1
C:\Python37\python.exe -m pytest -v --durations=50 --cov-report xml:tests_unittest.xml --cov-append tests\python\unittest\test_memory_opt.py
if ($LastExitCode -ne 0) { Throw ("Error running unittest, python exited with status code " + ('{0:X}' -f $LastExitCode)) }
