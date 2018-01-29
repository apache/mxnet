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
<var> = mx.sym.Convolution(data=<data>,
    <if(weight)>weight=<weight>,<endif>
    <if(bias)>bias=<bias>,<endif>
    kernel=(<kernel_h>,<kernel_w>),
    stride=(<stride_h>,<stride_w>),
    pad=(<pad_h>,<pad_w>),
    num_filter=<num_filter>,
    <if(no_bias)>no_bias=True,<endif>
    name='<name>')
