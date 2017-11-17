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

# train fcn-32s model
python -u fcn_xs.py --model=fcn32s --prefix=VGG_FC_ILSVRC_16_layers \
       --epoch=74 --init-type=vgg16

## train fcn-16s model
#python -u fcn_xs.py --model=fcn16s --prefix=FCN32s_VGG16 \
      #--epoch=31 --init-type=fcnxs

# train fcn-8s model
#python -u fcn_xs.py --model=fcn8s --prefix=FCN16s_VGG16 \
      #--epoch=27 --init-type=fcnxs
