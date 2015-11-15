Generating Random Sentence with LSTM RNN
========================================

This tutorial shows how to train a LSTM (Long short-term memory) RNN (recurrent
neural network) to perform character-level sequence training and prediction. The
original model, usually called ``char-rnn`` is described in `Andrej Karpathy's
blog <http://karpathy.github.io/2015/05/21/rnn-effectiveness/>`_, with
a reference implementation in Torch available `here
<https://github.com/karpathy/char-rnn>`_.

Because MXNet.jl does not have a specialized model for recurrent neural networks
yet, the example shown here is an implementation of LSTM by using the default
:class:`FeedForward` model via explicitly unfolding over time. We will be using
fixed-length input sequence for training. The code is adapted from the `char-rnn
example for MXNet's Python binding
<https://github.com/dmlc/mxnet/blob/master/example/rnn/char_lstm.ipynb>`_, which
demonstrates how to use low-level :doc:`symbolic APIs </api/symbolic-node>` to
build customized neural network models directly.

LSTM Cells
----------

Christopher Olah has a `great blog post about LSTM
<http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_ with beautiful and
clear illustrations. So we will not repeat the definition and explanation of
what an LSTM cell is here. Basically, an LSTM cell takes input ``x``, as well as
previous states (including ``c`` and ``h``), and produce the next states.
We define a helper type to bundle the two state variables together:

.. literalinclude:: ../../examples/char-lstm/lstm.jl
   :language: julia
   :start-after: #--LSTMState
   :end-before: #--/LSTMState

Because LSTM weights are shared at every time when we do explicit unfolding, so
we also define a helper type to hold all the weights (and bias) for an LSTM cell
for convenience.

.. literalinclude:: ../../examples/char-lstm/lstm.jl
   :language: julia
   :start-after: #--LSTMParam
   :end-before: #--/LSTMParam

Note all the variables are of type :class:`SymbolicNode`. We will construct the
LSTM network as a symbolic computation graph, which is then instantiated with
:class:`NDArray` for actual computation.

.. literalinclude:: ../../examples/char-lstm/lstm.jl
   :language: julia
   :start-after: #--lstm_cell
   :end-before: #--/lstm_cell

The following figure is stolen from
`Christopher Olah's blog
<http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_, which illustrate
exactly what the code snippet above is doing.

.. image:: http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png

In particular, instead of defining the four gates independently, we do the
computation together and then use :class:`SliceChannel` to split them into four
outputs. The computation of gates are all done with the symbolic API. The return
value is a LSTM state containing the output of a LSTM cell.

Unfolding LSTM
--------------
Using the LSTM cell defined above, we are now ready to define a function to
unfold a LSTM network with L layers and T time steps. The first part of the
function is just defining all the symbolic variables for the shared weights and
states.

.. literalinclude:: ../../examples/char-lstm/lstm.jl
   :language: julia
   :start-after: #--LSTM-part1
   :end-before: #--/LSTM-part1
