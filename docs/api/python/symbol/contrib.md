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

# Contrib Symbol API

```eval_rst
    .. currentmodule:: mxnet.symbol.contrib
```

## Overview

This document lists the contrib routines of the symbolic expression package:

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.symbol.contrib
```

The `Contrib Symbol` API, defined in the `symbol.contrib` package, provides
many useful experimental APIs for new features.
This is a place for the community to try out the new features,
so that feature contributors can receive feedback.

```eval_rst
.. warning:: This package contains experimental APIs and may change in the near future.
```

In the rest of this document, we list routines provided by the `symbol.contrib` package.

## Contrib

```eval_rst
.. currentmodule:: mxnet.symbol.contrib

.. autosummary::
    :nosignatures:

    AdaptiveAvgPooling2D
    BilinearResize2D
    CTCLoss
    DeformableConvolution
    DeformablePSROIPooling
    MultiBoxDetection
    MultiBoxPrior
    MultiBoxTarget
    MultiProposal
    PSROIPooling
    Proposal
    ROIAlign
    count_sketch
    ctc_loss
    dequantize
    fft
    ifft
    quantize
    foreach
    while_loop
    cond
    index_copy
    getnnz
    edge_id
    dgl_csr_neighbor_uniform_sample
    dgl_csr_neighbor_non_uniform_sample
    dgl_subgraph
    dgl_adjacency
    dgl_graph_compact
```

## API Reference

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst

.. automodule:: mxnet.symbol.contrib
    :members:

```

<script>auto_index("api-reference");</script>
