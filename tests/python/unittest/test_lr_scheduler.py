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

# pylint: skip-file
import mxnet as mx 
import mxnet.optimizer as opt              

def multi_lr_sceduler(lr, steps, lr_factor = 1, warmup_step = 0, warmup_lr = 0):
    lr_scheduler = None
    if warmup_step > 0 and warmup_lr > lr:
        lr_scheduler =  mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_factor, 
                    warmup_step = warmup_step, begin_lr=lr, stop_lr=warmup_lr)
    else:  
        lr_scheduler =  mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_factor) 

    optimizer_params = {
            'learning_rate': lr,
            'lr_scheduler': lr_scheduler}

    optimizer = opt.create('sgd', **optimizer_params)  
    updater = opt.get_updater(optimizer)     

    x = [[[[i*10+j for j in range(10)] for i in range(10)]]]
    x = mx.nd.array(x, dtype='float32')
    y = mx.nd.ones(shape = x.shape, dtype='float32') 

    res_lr = []
    for i in range(1,steps[-1] + 5):
        updater(0, y, x)
        cur_lr = optimizer._get_lr(0)
        res_lr.append(cur_lr)

    if warmup_step > 1:
        assert mx.test_utils.almost_equal(res_lr[warmup_step], warmup_lr, 1e-10) 
        lr = warmup_lr
    for i in range(len(steps)):
        assert mx.test_utils.almost_equal(res_lr[steps[i]], lr * pow(lr_factor, i + 1), 1e-10)  

def test_multi_lr_scheduler():
    #Legal input
    multi_lr_sceduler(lr = 0.02, steps=[100, 200])
    multi_lr_sceduler(lr = 0.2, steps = [8,12], lr_factor = 0.1, warmup_step = 0, warmup_lr = 0.1)
    multi_lr_sceduler(lr = 0.02, steps = [8,12], lr_factor = 0.1, warmup_step = 1, warmup_lr = 0.1)
    multi_lr_sceduler(lr = 0.02, steps = [8,12], lr_factor = 0.3, warmup_step = 5, warmup_lr = 0.1)
    multi_lr_sceduler(lr = 0.002, steps = [8,12], lr_factor = 0.1, warmup_step = 7, warmup_lr = 0.1)
    #Illegal input
    """
    #Schedule step must be greater than warmup_step
    multi_lr_sceduler(lr = 0.02, steps = [8,12], lr_factor = 0.1, warmup_step = 10, warmup_lr = 0.1)
    #stop_lr must larger than begin_lr
    multi_lr_sceduler(lr = 0.02, steps = [8,12], lr_factor = 0.1, warmup_step = 10, warmup_lr = 0.001)
    #Schedule step must be an list
    multi_lr_sceduler(lr = 0.02, steps = 8, lr_factor = 0.1, warmup_step = 5, warmup_lr = 0.1)
    #Factor must be no more than 1 to make lr reduce
    multi_lr_sceduler(lr = 0.02, steps = [8,12], lr_factor = 2, warmup_step = 5, warmup_lr = 0.1)
    #Schedule step must be an increasing integer list
    multi_lr_sceduler(lr = 0.02, steps = [12,8], lr_factor = 0.1, warmup_step = 5, warmup_lr = 0.1)
    """

if __name__ == "__main__":
    import nose
    nose.runmodule()
    
