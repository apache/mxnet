# Gluon Text API

## Overview

The `mxnet.gluon.text` APIs refer to classes and functions related to text data processing, such
as bulding indices and loading pre-trained embedding vectors for text tokens and storing them in the
`mxnet.ndarray.NDArray` format.

This document lists the text APIs in `mxnet.gluon`:

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.gluon.text.embedding
    mxnet.gluon.text.vocab
    mxnet.gluon.text.utils
```

All the code demonstrated in this document assumes that the following modules or packages are
imported.

```python
>>> from mxnet import gluon
>>> from mxnet import nd
>>> from mxnet.gluon import text
>>> import collections

```

### Indexing words and using pre-trained word embeddings in `gluon`

As a common use case, let us index words, attach pre-trained word embeddings for them, and use
such embeddings in `gluon` in just a few lines of code. 

To begin with, suppose that we have a simple text data set in the string format. We can count word
frequency in the data set.

```python
>>> text_data = " hello world \n hello nice world \n hi world \n"
>>> counter = text.count_tokens_from_str(text_data)

```

The obtained `counter` has key-value pairs whose keys are words and values are word frequencies.
This allows us to filter out infrequent words (See details at
[Vocabulary API specifications](#mxnet.gluon.text.vocab.Vocabulary)).
Suppose that we want to build indices for all the keys in `counter`. We need a Vocabulary instance
with `counter` as its argument.

```python
>>> my_vocab = text.Vocabulary(counter)

```

To attach word embedding to indexed words in `my_vocab`, let us go on to create a fastText word
embedding instance by specifying the embedding name `fasttext` and the pre-trained file name
`wiki.simple.vec`.

```python
>>> fasttext = text.embedding.create('fasttext', file_name='wiki.simple.vec')

```

So we can attach word embedding `fasttext` to indexed words `my_vocab`.

```python
>>> my_vocab.set_embedding(fasttext)

```

Now we are ready to access the fastText word embedding vectors for indexed words, such as 'hello'
and 'world'.

```python
>>> my_vocab.embedding[['hello', 'world']]

[[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
    ...
   -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
 [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
    ...
   -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
<NDArray 2x300 @cpu(0)>

```

To demonstrate how to use pre-trained word embeddings in the `gluon` package, let us first obtain
indices of the words 'hello' and 'world'.

```python
>>> my_vocab[['hello', 'world']]
[2, 1]

```

We can obtain the vector representation for the words 'hello' and 'world' by specifying their
indices (2 and 1) and the weight matrix `my_vocab.embedding.idx_to_vec` in
`mxnet.gluon.nn.Embedding`.
 
```python
>>> input_dim, output_dim = my_vocab.embedding.idx_to_vec.shape
>>> layer = gluon.nn.Embedding(input_dim, output_dim)
>>> layer.initialize()
>>> layer.weight.set_data(my_vocab.embedding.idx_to_vec)
>>> layer(nd.array([2, 1]))

[[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
    ...
   -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
 [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
    ...
   -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
<NDArray 2x300 @cpu(0)>

```

## Vocabulary

The vocabulary builds indices for text tokens and can be attached with token embeddings. The input
counter whose keys are candidate indices may be obtained via
[`count_tokens_from_str`](#mxnet.gluon.text.utils.count_tokens_from_str).


```eval_rst
.. currentmodule:: mxnet.gluon.text.vocab
.. autosummary::
    :nosignatures:

    Vocabulary
```

Suppose that we have a simple text data set in the string format. We can count word frequency in the
data set.

```python
>>> text_data = " hello world \n hello nice world \n hi world \n"
>>> counter = text.utils.count_tokens_from_str(text_data)

```

The obtained `counter` has key-value pairs whose keys are words and values are word frequencies.
This allows us to filter out infrequent words. Suppose that we want to build indices for the 2 most
frequent keys in `counter` with the unknown token representation '(unk)' and a reserved token
'(pad)'.

```python
>>> my_vocab = text.Vocabulary(counter, max_size=2, unknown_token='(unk)', 
...     reserved_tokens=['(pad)'])

```

We can access properties such as `token_to_idx` (mapping tokens to indices), `idx_to_token` (mapping
indices to tokens), `unknown_token` (representation of any unknown token) and `reserved_tokens`
(reserved tokens).


```python
>>> my_vocab.token_to_idx
{'(unk)': 0, '(pad)': 1, 'world': 2, 'hello': 3}
>>> my_vocab.idx_to_token
['(unk)', '(pad)', 'world', 'hello']
>>> my_vocab.unknown_token
'(unk)'
>>> my_vocab.reserved_tokens
['(pad)']
>>> len(my_vocab)
4
>>> my_vocab[['hello', 'world']]
[3, 2]
```

Besides the specified unknown token '(unk)' and reserved_token '(pad)' are indexed, the 2 most
frequent words 'world' and 'hello' are also indexed.


### Attach token embedding to vocabulary

A vocabulary instance can be attached with token embedding. 

To begin with, suppose that we have a simple text data set in the string format. We can count word
frequency in the data set.

```python
>>> text_data = " hello world \n hello nice world \n hi world \n"
>>> counter = text.count_tokens_from_str(text_data)

```

The obtained `counter` has key-value pairs whose keys are words and values are word frequencies.
This allows us to filter out infrequent words.
Suppose that we want to build indices for the most frequent 2 keys in `counter`. 

```python
>>> my_vocab = text.Vocabulary(counter, max_size=2)

```

Let us define the fastText word embedding instance with the pre-trained file `wiki.simple.vec`.

```python
>>> fasttext = text.embedding.create('fasttext', file_name='wiki.simple.vec')

```

So we can attach word embedding `fasttext` to indexed words `my_vocab`.

```python
>>> my_vocab.set_embedding(fasttext)

```

Now we are ready to access the fastText word embedding vectors for the indexed words.

```python
>>> my_vocab.embedding[['hello', 'world']]

[[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
    ...
   -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
 [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
    ...
   -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
<NDArray 2x300 @cpu(0)>

```

Let us define the GloVe word embedding with the pre-trained file `glove.6B.50d.txt`. Then, 
we can re-attach a GloVe text embedding instance to the vocabulary. 

```python
>>> glove = text.embedding.create('glove', file_name='glove.6B.50d.txt')
>>> my_vocab.set_embedding(glove)

```

Now we are ready to access the GloVe word embedding vectors for the indexed words.

```python
>>> my_vocab.embedding[['hello', 'world']]

[[  -0.38497001  0.80092001
    ...
    0.048833    0.67203999]
 [  -0.41486001  0.71847999
    ...
   -0.37639001 -0.67541999]]
<NDArray 2x50 @cpu(0)>

```

If a token is unknown to `my_vocab`, its embedding vector is initialized according to the default
specification in `glove` (all elements are 0).

```python

>>> my_vocab.embedding['nice']

[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  ...
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
<NDArray 50 @cpu(0)>

```



## Text token embedding

To load token embeddings from an externally hosted pre-trained token embedding file, such as those
of GloVe and FastText, use
[`embedding.create(embedding_name, file_name)`](#mxnet.gluon.text.embedding.create).

To get all the available `embedding_name` and `file_name`, use
[`embedding.get_file_names()`](#mxnet.gluon.text.embedding.get_file_names).

```python
>>> text.embedding.get_file_names()
{'glove': ['glove.42B.300d.txt', 'glove.6B.50d.txt', 'glove.6B.100d.txt', ...],
'fasttext': ['wiki.en.vec', 'wiki.simple.vec', 'wiki.zh.vec', ...]}

```

Alternatively, to load embedding vectors from a custom pre-trained text token embedding file, use
[`TokenEmbedding.from_file`](#mxnet.gluon.text.embedding.TokenEmbedding.from_file).


```eval_rst
.. currentmodule:: mxnet.gluon.text.embedding
.. autosummary::
    :nosignatures:

    register
    create
    get_file_names
    TokenEmbedding
    GloVe
    FastText
```

See [Assign token embedding to vocabulary](#assign-token-embedding-to-vocabulary) for how to attach
token embeddings to vocabulary and use token embeddings.


### Implement a new text token embedding

For ``embedding``, create a subclass of `mxnet.gluon.text.embedding.TokenEmbedding`.
Also add ``@mxnet.gluon.text.embedding.TokenEmbedding.register`` before this class. See
[`embedding.py`](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/text/embedding.py)
for examples.


## Text utilities

The following functions provide utilities for text data processing.

```eval_rst
.. currentmodule:: mxnet.gluon.text.utils
.. autosummary::
    :nosignatures:

    count_tokens_from_str
```


## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst

.. automodule:: mxnet.gluon.text.embedding
    :members: register, create, get_file_names
.. autoclass:: mxnet.gluon.text.embedding.TokenEmbedding
    :members: from_file
.. autoclass:: mxnet.gluon.text.embedding.GloVe
.. autoclass:: mxnet.gluon.text.embedding.FastText

.. automodule:: mxnet.gluon.text.vocab
.. autoclass:: mxnet.gluon.text.vocab.Vocabulary
    :members: set_embedding, to_tokens

.. automodule:: mxnet.gluon.text.utils
    :members: count_tokens_from_str

```
<script>auto_index("api-reference");</script>