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

# Executor and Executor Manager

The executor and executor manager are internal classes for managing symbolic
graph execution. This document is only intended for reference for advanced users.

.. note:: Direct interactions with executor and executor manager are dangerous and not recommended.

## Executor

```eval_rst
.. currentmodule:: mxnet.executor

.. autosummary::
    :nosignatures:

    Executor
```

## Executor Manager

```eval_rst
.. currentmodule:: mxnet.executor_manager

.. autosummary::
    :nosignatures:

    DataParallelExecutorGroup
    DataParallelExecutorManager
```

## API Reference

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.executor
    :members:
.. automodule:: mxnet.executor_manager
    :members:
```

<script>auto_index("api-reference");</script>
