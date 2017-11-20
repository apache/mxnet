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

#!/bin/python
# -*- coding: utf-8 -*-

'''
generate a summary of train log, including terms as below:
1. [2 terms] epoch-index, batch-num(at least)
2. [3 terms] ave-batch-train-acc(accuracy), max-batch-trian-acc, min-batch-train-acc,
3. [3 terms] ave-batch-train-topk-acc, max-batch-train-topk-acc, min-batch-train-topk-acc,
4. [2 terms] val-acc, val-topk-acc,
5. [3 terms] train-speed(image/second), max-speed, min-speed,
6. [2 terms] consume-time(seconds for current epoch), checkpoint name.
'''

# generate header first
header_list = ["epoch", "batchNum",
               "ave-batch-acc", "max-batch-acc", "min-batch-acc",
               "ave-batch-topk-acc", "max-batch-topk-acc", "min-batch-topk-acc",
               "train-acc", "train-topk-acc",
               "val-acc", "val-topk-acc",
               "train-speed(img/sec)", "max-speed", "min-speed",
               "consume-time(sec)", "checkpoint"]
header = "\t".join(header_list)

debug = False#True

def init():
    """
    1. define usage
    2. initialize parameters: define regular expressions for different terms.
    """
    global log_dir
    import sys

    try:
        log_dir = sys.argv[1]
    except IndexError:
        print("IndexError: please input log dir")
        print("".join(["Usage: python ", sys.argv[0], " YOUR_MXNET_LOG_DIRECTORY"]))
        exit(-1)

    global valid_line_pattern_list
    valid_line_pattern_list = ["Epoch", "Saved"]

    global epoch_pattern
    global batch_pattern
    global speed_pattern
    global consume_time_pattern
    global train_acc_pattern
    global train_topk_acc_pattern
    global val_acc_pattern
    global val_topk_acc_pattern
    global checkpoint_pattern

    epoch_pattern = ".*Epoch\[(.*?)\].*"
    batch_pattern = ".*Batch.*\[(.*?)\].*Speed:.*"
    speed_pattern = ".*Speed: (.*?) samples/sec.*"
    consume_time_pattern = ".*Time cost=(.*).*"
    train_acc_pattern = ".*Train-accuracy=(.*).*"
    train_topk_acc_pattern = ".*Train-top_k_accuracy_.*=(.*).*"
    val_acc_pattern = ".*] Validation-accuracy=(.*).*"
    val_topk_acc_pattern = ".*Validation-top_k_accuracy_.*=(.*).*"
    checkpoint_pattern = '.*Saved checkpoint to "(.*)".*'

def run():
    """
    1. read train log according to usage;
    2. match terms line by line according to regular expressions.
    """
    speed_list = []
    batch_list = []
    batch_train_acc_list = []
    batch_train_topk_acc_list = []
    train_topk_acc = "---"
    val_topk_acc = "---"
    line_idx = 0

    print(header)

    from re import findall
    with open(log_dir, "r") as log_handle:
        line = log_handle.readline()
        while line:
            line_idx += 1
            if debug: 
                if line_idx == 105: exit(-1)
            # check valid line
            valid_flag_list = filter(lambda ptn: ptn in line, valid_line_pattern_list)
            if debug: print "valid_flag_list:", valid_flag_list
            if len(valid_flag_list) < 1:
                if debug: print("invalid line")
                line = log_handle.readline()
                continue

            if debug: print "line " + str(line_idx) + "\t" + line
            train_acc_res = findall(train_acc_pattern, line)
            train_topk_acc_res = findall(train_topk_acc_pattern, line)
            val_acc_res = findall(val_acc_pattern, line)
            val_topk_acc_res = findall(val_topk_acc_pattern, line)

            epoch_res = findall(epoch_pattern, line)
            batch_res = findall(batch_pattern, line)
            speed_res = findall(speed_pattern, line)

            consume_time_res = findall(consume_time_pattern, line)
            checkpoint_res = findall(checkpoint_pattern, line)
            # consume time
            if len(consume_time_res) > 0:
                consume_time = consume_time_res[0]
            # checkpoint
            if len(checkpoint_res) > 0:
                checkpoint = checkpoint_res[0]
            # epoch
            if len(epoch_res) > 0: epoch = str(int(epoch_res[0]) + 1)
            # batch
            if len(batch_res) > 0: 
                batch = batch_res[0]
                batch_list.append(batch)
            # speed
            if len(speed_res) > 0: 
                speed = speed_res[0]
                speed_list.append(speed)

            # batch-train-acc
            if len(epoch_res) > 0 and len(train_acc_res) > 0 and len(batch_res) > 0:
                batch_train_acc = train_acc_res[0]
                batch_train_acc_list.append(batch_train_acc)
            # batch-train-topk-acc
            if len(epoch_res) > 0 and len(train_topk_acc_res) > 0 and len(batch_res) > 0:
                batch_train_topk_acc = train_topk_acc_res[0]
                batch_train_topk_acc_list.append(batch_train_topk_acc)

            # train-acc
            if len(epoch_res) > 0 and len(train_acc_res) > 0 and len(batch_res) < 1:
                train_acc = train_acc_res[0]
            # train-topk-acc
            if len(epoch_res) > 0 and len(train_topk_acc_res) > 0 and len(batch_res) < 1:
                train_topk_acc = train_topk_acc_res[0]
            # val-acc
            if len(epoch_res) > 0 and len(val_acc_res) > 0 and len(batch_res) < 1:
                val_acc = val_acc_res[0]
            # val-topk-acc
            if len(epoch_res) > 0 and len(val_topk_acc_res) > 0:
                val_topk_acc = val_topk_acc_res[0]
               
            # Saved line (checkpoint line)
            if len(batch_train_topk_acc_list) > 0:
                epoch_end_pattern = "Validation-top_k_accuracy"
            else:
                epoch_end_pattern = "Validation-accuracy"

            if epoch_end_pattern in line:
                # summarize
                if len(speed_list) != 0:
                    max_speed = max(speed_list)
                    min_speed = min(speed_list)
                    ave_speed = sum(map(float, speed_list))/float(len(speed_list))
                else:
                    max_speed = "---"
                    min_speed = "---"
                    ave_speed = "---"

                if len(batch_train_acc_list) != 0:
                    max_batch_train_acc = max(batch_train_acc_list)
                    min_batch_train_acc = min(batch_train_acc_list)
                    ave_batch_train_acc = sum(map(float, batch_train_acc_list))/float(len(batch_train_acc_list))
                else:
                    max_batch_train_acc = "---"
                    min_batch_train_acc = "---"
                    ave_batch_train_acc = "---"

                if len(batch_train_topk_acc_list) < 1:
                    max_batch_train_topk_acc = "---"
                    min_batch_train_topk_acc = "---"
                    ave_batch_train_topk_acc = "---"
                else:    
                    max_batch_train_topk_acc = max(batch_train_topk_acc_list)
                    min_batch_train_topk_acc = min(batch_train_topk_acc_list)
                    ave_batch_train_topk_acc = sum(map(float, batch_train_topk_acc_list)) / float(len(batch_train_topk_acc_list))
                if debug: print(val_topk_acc_res, line)
                print("{epoch}\t{batch}\t{ave_batch_train_acc}\t{max_batch_train_acc}\t{min_batch_train_acc}\t{ave_batch_train_topk_acc}\t{max_batch_train_topk_acc}\t{min_batch_train_topk_acc}\t{train_acc}\t{train_topk_acc}\t{val_acc}\t{val_topk_acc}\t{ave_speed}\t{max_speed}\t{min_speed}\t{consume_time}\t{checkpoint}".\
                      format(epoch=epoch,\
                             batch=batch,\
                             ave_batch_train_acc=ave_batch_train_acc,\
                             max_batch_train_acc=max_batch_train_acc,\
                             min_batch_train_acc=min_batch_train_acc,\
                             ave_batch_train_topk_acc=ave_batch_train_topk_acc,\
                             max_batch_train_topk_acc=max_batch_train_topk_acc,\
                             min_batch_train_topk_acc=min_batch_train_topk_acc,\
                             train_acc=train_acc,\
                             train_topk_acc=train_topk_acc,\
                             val_acc=val_acc,\
                             val_topk_acc=val_topk_acc,\
                             ave_speed=ave_speed,\
                             max_speed=max_speed,\
                             min_speed=min_speed,\
                             consume_time=consume_time,\
                             checkpoint=checkpoint)\
                )
                """
                print "batch:", batch,

                print "ave_batch_train_acc", ave_batch_train_acc,
                print "max_batch_train-acc", max_batch_train_acc,
                print "min_batch_train_acc", min_batch_train_acc,

                print "ave-batch-train-topk_acc", ave_batch_train_topk_acc,
                print "max_batch_train_topk_acc", max_batch_train_topk_acc,
                print "min_batch_train_topk_acc", min_batch_train_topk_acc,
   
                print "train-acc", train_acc,
                print "train-topk-acc:", train_topk_acc,
                print "val-acc", val_acc,
                print "val-topk-acc", val_topk_acc,

                print "ave_speed:", ave_speed,
                print "max_speed:", max_speed,
                print "min_speed:", min_speed,

                print "consume_time:", consume_time,
                print "checkpoint:", checkpoint
                """
                batch_train_acc_list = []
                batch_train_topk_acc_list = []
                speed_list = []
                batch_list = []
            line = log_handle.readline()


if __name__ == "__main__":
    init()
    run()
