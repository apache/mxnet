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
#!/usr/bin/env python3

import sys

from flake8.main import git

if __name__ == '__main__':
    sys.exit(
        git.hook(
            # (optional):
            # any value > 0 enables complexity checking with mccabe
            complexity=0,

            # (optional):
            # if True, this returns the total number of error which will cause
            # the hook to fail
            strict=True,

            # (optional):
            # a comma-separated list of errors and warnings to ignore
            # ignore=

            # (optional):
            # allows for the instances where you don't add the the files to the
            # index before running a commit, e.g git commit -a
            lazy=git.config_for('lazy'),
        )
    )
