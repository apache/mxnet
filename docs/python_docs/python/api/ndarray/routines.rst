Routines
========

Create Arrays
-------------

.. currentmodule:: mxnet.ndarray

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    arange
    array
    diag
    empty
    full
    load
    ones
    ones_like
    save
    zeros
    zeros_like


Manipulate
------------

Change shape and type
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    cast
    flatten
    expand_dims
    reshape
    reshape_like
    shape_array
    size_array


Expand  elements
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    broadcast_axes
    broadcast_like
    broadcast_to
    pad
    repeat
    tile


Rearrange elements
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    depth_to_space
    flip
    space_to_depth
    swapaxes
    transpose


Join and split
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    concat
    split
    stack


Index
^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    batch_take
    one_hot
    pick
    ravel_multi_index
    slice
    slice_axis
    slice_like
    take
    unravel_index
    where


Sequence
^^^^^^^^

.. currentmodule:: mxnet.ndarray

.. autosummary::
    :toctree: _autogen

    SequenceLast
    SequenceMask
    SequenceReverse


Math
----

Arithmetic
^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    add
    add_n
    batch_dot
    divide
    dot
    modulo
    multiply
    negative
    subtract


Trigonometric
^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    arccos
    arcsin
    arctan
    broadcast_hypot
    degrees
    cos
    radians
    sin
    tan


Hyperbolic
^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    arcsinh
    arccosh
    arctanh
    cosh
    sinh
    tanh


Reduce
^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    max
    min
    mean
    nanprod
    nansum
    norm
    prod
    sum


Round
^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    ceil
    fix
    floor
    round
    rint
    trunc


Exponents and logarithms
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    exp
    expm1
    log
    log1p
    log10
    log2


Powers
^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    cbrt
    power
    rcbrt
    reciprocal
    rsqrt
    square
    sqrt


Compare
^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    equal
    greater
    greater_equal
    lesser
    lesser_equal
    not_equal


Logical
^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    logical_and
    logical_not
    logical_or
    logical_xor


Sort and Search
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autogen

    argmax
    argmin
    argsort
    sort
    topk


Random Distribution
^^^^^^^^^^^^^^^^^^^^


.. autosummary::
    :nosignatures:
    :toctree: _autogen

    random.exponential
    random.gamma
    random.generalized_negative_binomial
    random.multinomial
    random.negative_binomial
    random.normal
    random.poisson
    random.randint
    random.randn
    random.shuffle
    random.uniform

Linear Algebra
^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    linalg.gelqf
    linalg.gemm
    linalg.gemm2
    linalg.potrf
    linalg.potri
    linalg.sumlogdiag
    linalg.syevd
    linalg.syrk
    linalg.trmm
    linalg.trsm


Miscellaneous
-------------

.. autosummary::
    :toctree: _autogen

    abs
    clip
    gamma
    gammaln
    maximum
    minimum
    sign


Neural Network
--------------

.. autosummary::
    :toctree: _autogen

    Activation
    BatchNorm
    BilinearSampler
    BlockGrad
    Convolution
    Correlation
    Custom
    Deconvolution
    Dropout
    Embedding
    FullyConnected
    GridGenerator
    IdentityAttachKLSparseReg
    InstanceNorm
    L2Normalization
    LayerNorm
    LeakyReLU
    LinearRegressionOutput
    log_softmax
    LogisticRegressionOutput
    LRN
    MAERegressionOutput
    MakeLoss
    Pooling
    relu
    ROIPooling
    RNN
    sigmoid
    smooth_l1
    softmax
    softmax_cross_entropy
    SoftmaxOutput
    SoftmaxActivation
    SpatialTransformer
    SVMOutput
    UpSampling

Contributed routines
--------------------

.. automodule:: mxnet.ndarray.contrib

.. note:: This package contains experimental APIs and may change in the near future.

Manipulate
^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    count_sketch
    getnnz
    index_copy


FFT
^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    fft
    ifft


Quantization
^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    dequantize
    quantize


Neural network
^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    AdaptiveAvgPooling2D
    BilinearResize2D
    ctc_loss
    DeformableConvolution
    DeformablePSROIPooling
    MultiBoxDetection
    MultiBoxPrior
    MultiBoxTarget
    MultiProposal
    Proposal
    PSROIPooling
    ROIAlign


Control flow
^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    cond
    foreach
    while_loop