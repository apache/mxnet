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

"""MovieLens data handling: download, parse, and expose as DataIter
"""

import os
import mxnet as mx

def load_mldata_iter(filename, batch_size):
    """Not particularly fast code to parse the text file and load it into three NDArray's
    and product an NDArrayIter
    """
    user = []
    item = []
    score = []
    with open(filename) as f:
        for line in f:
            tks = line.strip().split('\t')
            if len(tks) != 4:
                continue
            user.append(int(tks[0]))
            item.append(int(tks[1]))
            score.append(float(tks[2]))
    user = mx.nd.array(user)
    item = mx.nd.array(item)
    score = mx.nd.array(score)
    return mx.io.NDArrayIter(data={'user':user,'item':item},label={'score':score},
                             batch_size=batch_size, shuffle=True)

def ensure_local_data(prefix):
    if not os.path.exists("%s.zip" % prefix):
        print("Downloading MovieLens data: %s" % prefix)
        os.system("wget http://files.grouplens.org/datasets/movielens/%s.zip" % prefix)
        os.system("unzip %s.zip" % prefix)


def get_data_iter(batch_size, prefix='ml-100k'):
    """Returns a pair of NDArrayDataIter, one for train, one for test.
    """
    ensure_local_data(prefix)
    return (load_mldata_iter('./%s/u1.base' % prefix, batch_size),
            load_mldata_iter('./%s/u1.test' % prefix, batch_size))

def max_id(fname):
    mu = 0
    mi = 0
    for line in open(fname):
        tks = line.strip().split('\t')
        if len(tks) != 4:
            continue
        mu = max(mu, int(tks[0]))
        mi = max(mi, int(tks[1]))
    return mu + 1, mi + 1
