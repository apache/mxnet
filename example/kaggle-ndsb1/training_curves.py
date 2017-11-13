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

## based on https://github.com/dmlc/mxnet/issues/1302
## Parses the model fit log file and generates a train/val vs epoch plot
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Parses log file and generates train/val curves')
parser.add_argument('--log-file', type=str,default="log_tr_va",
                    help='the path of log file')
args = parser.parse_args()


TR_RE = re.compile('.*?]\sTrain-accuracy=([\d\.]+)')
VA_RE = re.compile('.*?]\sValidation-accuracy=([\d\.]+)')

log = open(args.log_file).read()

log_tr = [float(x) for x in TR_RE.findall(log)]
log_va = [float(x) for x in VA_RE.findall(log)]
idx = np.arange(len(log_tr))

plt.figure(figsize=(8, 6))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(idx, log_tr, 'o', linestyle='-', color="r",
         label="Train accuracy")

plt.plot(idx, log_va, 'o', linestyle='-', color="b",
         label="Validation accuracy")

plt.legend(loc="best")
plt.xticks(np.arange(min(idx), max(idx)+1, 5))
plt.yticks(np.arange(0, 1, 0.2))
plt.ylim([0,1])
plt.show()
