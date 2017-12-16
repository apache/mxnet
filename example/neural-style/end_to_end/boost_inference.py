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
import mxnet as mx
import numpy as np

#import basic
import data_processing
import gen_v3
import gen_v4

dshape = (1, 3, 480, 640)
clip_norm = 1.0 * np.prod(dshape)
model_prefix = "./model/"
ctx = mx.gpu(0)

# generator
gens = [gen_v4.get_module("g0", dshape, ctx),
        gen_v3.get_module("g1", dshape, ctx),
        gen_v3.get_module("g2", dshape, ctx),
        gen_v4.get_module("g3", dshape, ctx)]
for i in range(len(gens)):
    gens[i].load_params("./model/%d/v3_0002-0026000.params" % i)

content_np = data_processing.PreprocessContentImage("../input/IMG_4343.jpg", min(dshape[2:]), dshape)
data = [mx.nd.array(content_np)]
for i in range(len(gens)):
    gens[i].forward(mx.io.DataBatch([data[-1]], [0]), is_train=False)
    new_img = gens[i].get_outputs()[0]
    data.append(new_img.copyto(mx.cpu()))
    data_processing.SaveImage(new_img.asnumpy(), "out_%d.jpg" % i)
