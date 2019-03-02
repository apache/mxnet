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

# MXNet Architecture

Building a high-performance deep learning library
requires many systems-level design decisions.
In this design note, we share the rationale
for the specific choices made when designing _MXNet_.
We imagine that these insights may be useful
to both deep learning practitioners
and builders of other deep learning systems.

## Deep Learning System Design Concepts

The following pages address general design concepts for deep learning systems.
Mainly, they focus on the following 3 areas:
abstraction, optimization, and trade-offs between efficiency and flexibility.
Additionally, we provide an overview of the complete MXNet system.

```eval_rst
.. toctree::
   :maxdepth: 1

   overview.md
   program_model.md
   note_engine.md
   note_memory.md
   note_data_loading.md
   exception_handling.md
   rnn_interface.md
```
