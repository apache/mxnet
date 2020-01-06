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

# Bidirectionnal LSTM to sort an array.

This is an example of using bidirectionmal lstm to sort an array. Please refer to the notebook.

We train a bidirectionnal LSTM to sort an array of integer.

For example:

`500 30 999 10 130` should give us `10 30 130 500 999`

![](https://cdn-images-1.medium.com/max/1200/1*6QnPUSv_t9BY9Fv8_aLb-Q.png)


([Diagram source](http://colah.github.io/posts/2015-09-NN-Types-FP/))