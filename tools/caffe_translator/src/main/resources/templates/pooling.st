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
<var> = mx.symbol.Pooling(data=<data>,
    pool_type='<type>',
<if(global_pool)>
    global_pool=<global_pool>,
<endif>
<if(kernel_h)>
    kernel=(<kernel_h>,<kernel_w>),
<endif>
<if(stride_h)>
    stride=(<stride_h>,<stride_w>),
<endif>
<if(pad_h)>
    pad=(<pad_h>,<pad_w>),
<endif>
    pooling_convention='full',
    name='<name>')
