<!
 Licensed to the Apache Software Foundation (ASF) under one
 or more contributor license agreements.  See the NOTICE file
 distributed with this work for additional information
 regarding copyright ownership.  The ASF licenses this file
 to you under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance
 with the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an
 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 KIND, either express or implied.  See the License for the
 specific language governing permissions and limitations
 under the License.
!>
ctx = <ctx>

module = mx.mod.Module(symbol=<loss>, context=ctx, data_names=[<data_names;separator=", ">], label_names=[<label_names;separator=", ">])
module.bind(data_shapes=<train_data_itr>.provide_data,
            label_shapes=<train_data_itr>.provide_label)

<init_params>

<init_optimizer>

epoch = 1
batch_num = 1

max_iter = <max_iter>
snapshot = <snapshot>
test_interval = <test_interval>
test_iter = <test_iter>

while batch_num \<= max_iter:
    <train_data_itr>.reset()

    for batch in <train_data_itr>:
        module.forward(data_batch=batch, is_train=True)
        module.backward()
        module.update()

        train_metrics.process(batch_num, module, batch.label)

        if(batch_num % test_interval == 0):
            logger.info("Iteration %d, Testing net" % batch_num)
            test_metrics.score_and_print(module, <test_data_itr>, num_batch=test_iter)

        if(batch_num % snapshot == 0):
            # write snapshot
            module.save_checkpoint(prefix="<snapshot_prefix>", epoch=batch_num, save_optimizer_states=True)

        batch_num += 1

        if batch_num > max_iter:
            break

<if(stepsize)>
        stepsize = <stepsize>
<endif>
<if(gamma)>
        gamma = <gamma>
<endif>
<if(power)>
        power = <power>
<endif>
<lr_update>

    epoch += 1


logger.info("Training done. Saving model to <snapshot_prefix>")
module.save_checkpoint(prefix="<snapshot_prefix>", epoch=batch_num, save_optimizer_states=True)
