<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->
<!--- -->
<!---   http://www.apache.org/licenses/LICENSE-2.0 -->
<!--- -->
<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Symbol

This fold contains definition of various networks. To add a new network, please
use the following format.

## Python

- A file implements one network proposed in a paper, with the network name as the
filename.
- Mention the paper and the modifications made if any at the beginning
of the file.
- Indicate how to reproduce the accuracy numbers in the paper if it is not straightforward
- Provide a function `get_symbol()` that return the network
