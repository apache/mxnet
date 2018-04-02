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

import random

vocab = [str(x) for x in range(100, 1000)]
sw_train = open("sort.train.txt", "w")
sw_test = open("sort.test.txt", "w")
sw_valid = open("sort.valid.txt", "w")

for i in range(1000000):
    seq = " ".join([vocab[random.randint(0, len(vocab) - 1)] for j in range(5)])
    k = i % 50
    if k == 0:
        sw_test.write(seq + "\n")
    elif k == 1:
        sw_valid.write(seq + "\n")
    else:
        sw_train.write(seq + "\n")

sw_train.close()
sw_test.close()
sw_valid.close()
