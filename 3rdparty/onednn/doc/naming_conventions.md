Naming Conventions {#dev_guide_conventions}
===========================================

oneDNN documentation relies on a set of standard naming
conventions for variables. This section describes these conventions.

## Variable (Tensor) Names

Neural network models consist of operations of the following form:
\f[ \dst = f(\src, \weights), \f]
where \dst and \src are activation tensors, and \weights are
learnable tensors.

The backward propagation consists then in computing the gradients with respect
to the \src and \weights respectively:
\f[ \diffsrc = df_{\src}(\diffdst, \src, \weights, \dst), \f] and
\f[ \diffweights = df_{\weights}(\diffdst, \src, \weights, \dst). \f]

While oneDNN uses _src_, _dst_, and _weights_ as generic names for the
activations and learnable tensors, for a specific operation there might be
commonly used and widely known specific names for these tensors.
For instance, the [convolution](@ref dev_guide_convolution) operation has a
learnable tensor called bias. For usability reasons, oneDNN primitives
use such names in initialization or other functions to simplify the coding.

To summarize, oneDNN uses the following commonly used notations for
tensors:

| Name                  | Meaning
| :-                    | :-
| `src`                 | Source tensor
| `dst`                 | Destination tensor
| `weights`             | Weights tensor
| `bias`                | Bias tensor (used in @ref dev_guide_convolution, @ref dev_guide_inner_product and other primitives)
| `scale_shift`         | Scale and shift tensors (used in @ref dev_guide_batch_normalization and @ref dev_guide_layer_normalization)
| `workspace`           | Workspace tensor that carries additional information from the forward propagation to the backward propagation
| `scratchpad`          | Temporary tensor that is required to store the intermediate results
| `diff_src`            | Gradient tensor with respect to the source
| `diff_dst`            | Gradient tensor with respect to the destination
| `diff_weights`        | Gradient tensor with respect to the weights
| `diff_bias`           | Gradient tensor with respect to the bias
| `diff_scale_shift`    | Gradient tensor with respect to the scale and shift
| `*_layer`             | RNN layer data or weights tensors
| `*_iter`              | RNN recurrent data or weights tensors


## Formulas and Verbose Output

oneDNN uses the following notations in the documentation formulas and verbose
output. Here, lower-case letters are used to denote indices in a particular
spatial dimension, the sizes of which are denoted by corresponding upper-case
letters.

| Name                                     | Semantics
| :--------------------------------------- | :----------------------------------------
| `n` (or `mb`)                            | batch
| `g`                                      | groups
| `oc`, `od`, `oh`, `ow`                   | output channels, depth, height, and width
| `ic`, `id`, `ih`, `iw`                   | input channels, depth, height, and width
| `kd`, `kh`, `kw`                         | kernel (filter) depth, height, and width
| `sd`, `sh`, `sw`                         | stride by depth, height, and width
| `dd`, `dh`, `dw`                         | dilation by depth, height, and width
| `pd`, `ph`, `pw`                         | padding by depth, height, and width

## RNN-Specific Notation

The following notations are used when describing RNN primitives.

| Name            | Semantics
| :-------------- | :----------------------------------
| \f$\cdot\f$     | matrix multiply operator
| \f$ * \f$       | element-wise multiplication operator
| W               | input weights
| U               | recurrent weights
| \f$^T\f$        | transposition
| B               | bias
| h               | hidden state
| a               | intermediate value
| x               | input
| \f$ _t {}_{}\f$ | timestamp
| \f$ l \f$       | layer index
| activation      | tanh, relu, logistic
| c               | cell state
| \f$\tilde{c}\f$ | candidate state
| i               | input gate
| f               | forget gate
| o               | output gate
| u               | update gate
| r               | reset gate


## Memory Formats Tags

When describing tensor memory formats, which is the oneDNN term for the
way that the data is laid out in memory, documentation uses letters of the
English alphabet to describe an order of dimensions and their semantics.

The canonical sequence of letters is _a_, _b_, _c_, ..., _z_. In this notation,
the _ab_ tag denotes a two-dimensional tensor with _a_ denoting the outermost
dimension and _b_ denoting the innermost dimension, where the latter is dense in
memory. Further, the _ba_ tag denotes a two-dimensional tensor but with last two
dimensions transposed: instead of the naturally dense _b_ dimension, now _a_ is
the dense dimension. If we suppose that the two-dimensional tensor is a matrix
and the _a_ and _b_ dimensions represent the number of columns and rows, then
_ab_ would denote the row-major (C) format and _ba_ would denote the
column-major (Fortran) format.

@todo Picture here

Upper-case letters are used to indicate that the data is laid out in blocks for
a particular dimension. In such cases, the format name contains both upper- and
lower-case letters for that dimension with a lower-case letter preceded by the
block size. For example, the _Ab16a_ tag denotes a format similar to row-major
but with columns split into contiguous blocks of 16 elements each. Moreover, the
implicit assumption is that if the number of columns is not divisible by 16, the
last block in the in-memory representation will contain padding zeroes.

@todo Picture here

Since there are many widely used names for specific deep learning domains like
convolutional neural networks (CNNs), oneDNN also supports memory
format tags in which dimensions have specifically assigned meaning like 'image
width', 'image height', etc. The following table summarizes notations used in
such memory format tags.

| Letter  | Dimension                               |
| ------- | --------------------------------------- |
| n       | batch                                   |
| g       | groups                                  |
| c       | channels                                |
| o       | output channels                         |
| i       | input channels                          |
| h       | height                                  |
| w       | width                                   |
| d       | depth                                   |
| t       | timestamp                               |
| l       | layer                                   |
| d       | direction                               |
| g       | gate                                    |
| s       | state                                   |

The canonical sequence of dimensions for four-dimensional data tensors in CNNs
is (batch, channels, spatial dimensions). Spatial dimensions are ordered for
tensors with three spatial dimensions as (depth, height, width), for tensors
with two spatial dimensions as (height, width), and as just (width) for tensors
with only one spatial dimension.

In this notation, _nchw_ is a memory format tag for a four-dimensional tensor,
with the first dimension corresponding to batch, the second to channel, and
the remaining two to spatial dimensions. Due to the canonical order of
dimensions for CNNs, this tag is the same as _abcd_. As another example, _nhwc_
is the same as _acdb_.
