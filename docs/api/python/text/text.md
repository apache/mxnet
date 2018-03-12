# Text API

## Overview

The `mxnet.text` APIs refer to classes and functions related to text data processing, such
as bulding indices and loading pre-trained embedding vectors for text tokens and storing them in the
`mxnet.ndarray.NDArray` format.

This document lists the text APIs in mxnet:

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.text.embedding
    mxnet.text.vocab
    mxnet.text.utils
```

All the code demonstrated in this document assumes that the following modules or packages are
imported.

```python
>>> from mxnet import gluon
>>> from mxnet import nd
>>> from mxnet import text
>>> import collections

```

### Access pre-trained word embeddings for indexed words

As a common use case, let us access pre-trained word embedding vectors for indexed words in just a
few lines of code. 

To begin with, let us create a fastText word embedding instance by specifying the embedding name
`fasttext` and the pre-trained file `wiki.simple.vec`.

```python
>>> fasttext = text.embedding.create('fasttext', file_name='wiki.simple.vec')

```

Now, suppose that we have a simple text data set in the string format. We can count
word frequency in the data set.

```python
>>> text_data = " hello world \n hello nice world \n hi world \n"
>>> counter = text.utils.count_tokens_from_str(text_data)

```

The obtained `counter` has key-value pairs whose keys are words and values are word frequencies.
Suppose that we want to build indices for all the keys in `counter` and load the defined fastText
word embedding for all such indexed words. We need a Vocabulary object with `counter` and
`embedding` as its arguments

```python
>>> my_vocab = text.vocab.Vocabulary(counter, embedding=fasttext)

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

### Using pre-trained word embeddings in `gluon`

To demonstrate how to use pre-trained word embeddings in the `gluon` package, let us first obtain
indices of the words 'hello' and 'world'.

```python
>>> my_vocab[['hello', 'world']]
[2, 1]

```

We can obtain the vector representation for the words 'hello' and 'world' by specifying their
indices (2 and 1) and the `my_embedding.idx_to_vec` in `mxnet.gluon.nn.Embedding`.
 
```python
>>> layer = gluon.nn.Embedding(my_vocab.embedding.idx_to_vec.shape[0], my_vocab.embedding.idx_to_vec.shape[1])
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

The vocabulary builds indices for text tokens and can be assigned with token embeddings. The input
counter whose keys are candidate indices may be obtained via
[`count_tokens_from_str`](#mxnet.text.utils.count_tokens_from_str).


```eval_rst
.. currentmodule:: mxnet.text.vocab
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
Suppose that we want to build indices for the 2 most frequent keys in `counter` with the unknown
token representation '&lt;unk&gt;' and a reserved token '&lt;pad&gt;'.

```python
>>> my_vocab = text.vocab.Vocabulary(counter, max_size=2, unknown_token='(unk)', 
...     reserved_tokens=['(pad)'])

```

We can access properties such as `token_to_idx` (mapping tokens to indices), `idx_to_token` (mapping
indices to tokens), `unknown_token` (representation of any unknown token) and `reserved_tokens`.


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


### Assign token embedding to vocabulary

A vocabulary instance can be assigned with token embedding. 

To begin with, suppose that we have a simple text data set in the string format. We can count word
frequency in the data set.

```python
>>> text_data = " hello world \n hello nice world \n hi world \n"
>>> counter = text.utils.count_tokens_from_str(text_data)

```

Let us define the fastText word embedding with pre-trained file `wiki.simple.vec`.

```python
>>> fasttext = text.embedding.create('fasttext', file_name='wiki.simple.vec')

```

The obtained `counter` has key-value pairs whose keys are words and values are word frequencies.
Suppose that we want to build indices for the most frequent 2 keys in `counter` and load the defined
fastText word embedding for all these 2 words. 

```python
>>> my_vocab = text.vocab.Vocabulary(counter, max_size=2, embedding=fasttext)

```

Now we are ready to access the fastText word embedding vectors for indexed words.

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

Let us define the GloVe word embedding with pre-trained file `glove.6B.50d.txt`. Then, 
we can re-assign a GloVe text embedding to the vocabulary. 

```python
>>> glove = text.embedding.create('glove', file_name='glove.6B.50d.txt')
>>> my_vocab.set_embedding(glove)

```

Now we are ready to access the GloVe word embedding vectors for indexed words.

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

If a token is unknown to `glossary`, its embedding vector is initialized according to the default
specification in `fasttext_simple` (all elements are 0).

```python

>>> my_embedding.get_vecs_by_tokens('nice')

[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  ...
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
<NDArray 300 @cpu(0)>

```



## Text token embedding

To load token embeddings from an externally hosted pre-trained token embedding file, such as those
of GloVe and FastText, use
[`embedding.create(embedding_name, file_name)`](#mxnet.text.embedding.create).

To get all the available `embedding_name` and `pretrained_file_name`, use
[`embedding.get_file_names()`](#mxnet.text.embedding.get_file_names).

```python
>>> text.embedding.get_file_names()
{'glove': ['glove.42B.300d.txt', 'glove.6B.50d.txt', 'glove.6B.100d.txt', ...],
'fasttext': ['wiki.en.vec', 'wiki.simple.vec', 'wiki.zh.vec', ...]}

```

Alternatively, to load embedding vectors from a custom pre-trained text token
embedding file, use [`TokenEmbedding.from_file`](#mxnet.text.embedding.TokenEmbedding.from_file).


```eval_rst
.. currentmodule:: mxnet.text.embedding
.. autosummary::
    :nosignatures:

    register
    create
    get_file_names
    TokenEmbedding
    GloVe
    FastText
```





### Indexed tokens are from the loaded embedding vectors

One can also use all the tokens from the loaded embedding vectors, such as loaded from a pre-trained
token embedding file, as the indexed tokens of the embedding.

To begin with, we can create a fastText word embedding object by specifying the embedding name
'fasttext' and the pre-trained file 'wiki.simple.vec'. The argument `init_unknown_vec` specifies
default vector representation for any unknown token. To index all the tokens from this pre-trained
word embedding file, we do not need to specify any vocabulary.

```python
>>> my_embedding = text.embedding.create('fasttext', pretrained_file_name='wiki.simple.vec',
...     init_unknown_vec=nd.zeros)

```

We can access properties such as `token_to_idx` (mapping tokens to indices), `idx_to_token` (mapping
indices to tokens), `vec_len` (length of each embedding vector), and `unknown_token` (representation
of any unknown token, default value is '&lt;unk&gt;').

```python
>>> my_embedding.token_to_idx['nice']
2586
>>> my_embedding.idx_to_token[2586]
'nice'
>>> my_embedding.vec_len
300
>>> my_embedding.unknown_token
'&lt;unk&gt;'

```

For every unknown token, if its representation '&lt;unk&gt;' is encountered in the pre-trained token
embedding file, index 0 of property `idx_to_vec` maps to the pre-trained token embedding vector
loaded from the file; otherwise, index 0 of property `idx_to_vec` maps to the default token
embedding vector specified via `init_unknown_vec` (set to nd.zeros here). Since the pre-trained file
does not have a vector for the token '&lt;unk&gt;', index 0 has to map to an additional token '&lt;unk&gt;' and
the number of tokens in the embedding is 111,052.


```python
>>> len(my_embedding)
111052
>>> my_embedding.idx_to_vec[0]

[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  ...
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
<NDArray 300 @cpu(0)>
>>> my_embedding.get_vecs_by_tokens('nice')

[ 0.49397001  0.39996001  0.24000999 -0.15121    -0.087512    0.37114
  ...
  0.089521    0.29175001 -0.40917999 -0.089206   -0.1816     -0.36616999]
<NDArray 300 @cpu(0)>
>>> my_embedding.get_vecs_by_tokens(['unknownT0kEN', 'unknownT0kEN'])

[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
   ...
   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
   ...
   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
<NDArray 2x50 @cpu(0)>

```


### Implement a new text token embedding

For ``optimizer``, create a subclass of `mxnet.text.embedding.TokenEmbedding`.
Also add ``@mxnet.text.embedding.TokenEmbedding.register`` before this class. See
[`embedding.py`](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/text/embedding.py)
for examples.


## Text utilities

The following functions provide utilities for text data processing.

```eval_rst
.. currentmodule:: mxnet.text.utils
.. autosummary::
    :nosignatures:

    count_tokens_from_str
```


## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst

.. automodule:: mxnet.text.embedding
    :members: register, create, get_file_names
.. autoclass:: mxnet.text.embedding.TokenEmbedding
    :members: from_file
.. autoclass:: mxnet.text.embedding.GloVe
.. autoclass:: mxnet.text.embedding.FastText

.. automodule:: mxnet.text.vocab
.. autoclass:: mxnet.text.vocab.Vocabulary
    :members: set_embedding, to_tokens

.. automodule:: mxnet.text.utils
    :members: count_tokens_from_str

```
<script>auto_index("api-reference");</script>