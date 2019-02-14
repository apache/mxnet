<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Neural art

This is an implementation of the paper
[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon
A. Gatys, Alexander S. Ecker, and Matthias Bethge.

## How to use

First use `download.sh` to download pre-trained model and sample inputs

Then run `python nstyle.py`, use `-h` to see more options

## Sample results

<img src=https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/output/4343_starry_night.jpg width=600px>

It takes 30 secs for a Titan X to generate the above 600x400 image.

## Note

* The current implementation is based the
  [torch implementation](https://github.com/jcjohnson/neural-style). But we may
  change it dramatically in the near future.

* We will release multi-GPU version soon.
