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
import zipfile
import shutil
from mxnet.test_utils import download

zip_file_path = 'models/msgnet_21styles.zip'
download('https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/models/msgnet_21styles-2cb88353.zip', zip_file_path)

with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall()

os.remove(zip_file_path)

shutil.move('msgnet_21styles-2cb88353.params', 'models/21styles.params')
