# Text API

## Overview

The mxnet.contrib.text APIs refer to classes and functions related to text data
processing, such as bulding indices and loading pre-trained embedding vectors
for text tokens and storing them in the `mxnet.ndarray.NDArray` format.

```eval_rst
.. warning:: This package contains experimental APIs and may change in the near future.
```

This document lists the text APIs in mxnet:

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.contrib.text.glossary
    mxnet.contrib.text.embedding
    mxnet.contrib.text.indexer
    mxnet.contrib.text.utils
```

All the code demonstrated in this document assumes that the following modules
or packages are imported.

```python
>>> from mxnet import gluon
>>> from mxnet import nd
>>> from mxnet.contrib import text
>>> import collections

```

### Look up pre-trained word embeddings for indexed words

As a common use case, let us look up pre-trained word embedding vectors for
indexed words in just a few lines of code. To begin with, we can create a
fastText word embedding object by specifying the embedding name `fasttext` and
the pre-trained file `wiki.simple.vec`.

```python
>>> fasttext_simple = text.embedding.TokenEmbedding.create('fasttext',
...     pretrained_file_name='wiki.simple.vec')

```

Suppose that we have a simple text data set in the string format. We can count
word frequency in the data set.

```python
>>> text_data = " hello world \n hello nice world \n hi world \n"
>>> counter = text.utils.count_tokens_from_str(text_data)

```

The obtained `counter` has key-value pairs whose keys are words and values are
word frequencies. Suppose that we want to build indices for all the keys in
`counter` and load the defined fastText word embedding for all such indexed
words. First, we need a TokenIndexer object with `counter` as its argument

```python
>>> token_indexer = text.indexer.TokenIndexer(counter)

```

Then, we can create a Glossary object by specifying `token_indexer` and `fasttext_simple` as its
arguments.

```python
>>> glossary = text.glossary.Glossary(token_indexer, fasttext_simple)

```

Now we are ready to look up the fastText word embedding vectors for indexed
words.

```python
>>> glossary.get_vecs_by_tokens(['hello', 'world'])

[[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
    ...
   -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
 [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
    ...
   -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
<NDArray 2x300 @cpu(0)>

```

### Use `glossary` in `gluon`

To demonstrate how to use a glossary with the loaded word embedding in the
`gluon` package, let us first obtain indices of the words 'hello' and 'world'.

```python
>>> glossary.to_indices(['hello', 'world'])
[2, 1]

```

We can obtain the vector representation for the words 'hello' and 'world'
by specifying their indices (2 and 1) and the `glossary.idx_to_vec` in
`mxnet.gluon.nn.Embedding`.
 
```python
>>> layer = gluon.nn.Embedding(len(glossary), glossary.vec_len)
>>> layer.initialize()
>>> layer.weight.set_data(glossary.idx_to_vec)
>>> layer(nd.array([2, 1]))

[[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
    ...
   -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
 [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
    ...
   -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
<NDArray 2x300 @cpu(0)>

```


## Glossary

The glossary provides indexing and embedding for text tokens in a glossary. For
each indexed token in a glossary, an embedding vector will be associated with
it. Such embedding vectors can be loaded from externally hosted or custom
pre-trained token embedding files, such as via instances of
[`TokenEmbedding`](#mxnet.contrib.text.embedding.TokenEmbedding). 
The input counter whose keys are
candidate indices may be obtained via
[`count_tokens_from_str`](#mxnet.contrib.text.utils.count_tokens_from_str).

```eval_rst
.. currentmodule:: mxnet.contrib.text.glossary
.. autosummary::
    :nosignatures:

    Glossary
```

To get all the valid names for pre-trained embeddings and files, we can use
[`TokenEmbedding.get_embedding_and_pretrained_file_names`](#mxnet.contrib.text.embedding.TokenEmbedding.get_embedding_and_pretrained_file_names).

```python
>>> text.embedding.TokenEmbedding.get_embedding_and_pretrained_file_names()
{'glove': ['glove.42B.300d.txt', 'glove.6B.50d.txt', 'glove.6B.100d.txt',
'glove.6B.200d.txt', 'glove.6B.300d.txt', 'glove.840B.300d.txt',
'glove.twitter.27B.25d.txt', 'glove.twitter.27B.50d.txt',
'glove.twitter.27B.100d.txt', 'glove.twitter.27B.200d.txt'],
'fasttext': ['wiki.en.vec', 'wiki.simple.vec', 'wiki.zh.vec']}

```

To begin with, we can create a fastText word embedding object by specifying the
embedding name `fasttext` and the pre-trained file `wiki.simple.vec`.

```python
>>> fasttext_simple = text.embedding.TokenEmbedding.create('fasttext',
...     pretrained_file_name='wiki.simple.vec')

```

Suppose that we have a simple text data set in the string format. We can count
word frequency in the data set.

```python
>>> text_data = " hello world \n hello nice world \n hi world \n"
>>> counter = text.utils.count_tokens_from_str(text_data)

```

The obtained `counter` has key-value pairs whose keys are words and values are
word frequencies. Suppose that we want to build indices for the most frequent 2
keys in `counter` and load the defined fastText word embedding for all these
2 words. 

```python
>>> token_indexer = text.indexer.TokenIndexer(counter, most_freq_count=2)
>>> glossary = text.glossary.Glossary(token_indexer, fasttext_simple)

```

Now we are ready to look up the fastText word embedding vectors for indexed
words.

```python
>>> glossary.get_vecs_by_tokens(['hello', 'world'])

[[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
    ...
   -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
 [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
    ...
   -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
<NDArray 2x300 @cpu(0)>

```

We can also access properties such as `token_to_idx` (mapping tokens to
indices), `idx_to_token` (mapping indices to tokens), and `vec_len`
(length of each embedding vector).

```python
>>> glossary.token_to_idx
{'<unk>': 0, 'world': 1, 'hello': 2, 'hi': 3, 'nice': 4}
>>> glossary.idx_to_token
['<unk>', 'world', 'hello', 'hi', 'nice']
>>> len(glossary)
5
>>> glossary.vec_len
300

```

If a token is unknown to `glossary`, its embedding vector is initialized
according to the default specification in `fasttext_simple` (all elements are
0).

```python

>>> glossary.get_vecs_by_tokens('unknownT0kEN')

[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  ...
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
<NDArray 300 @cpu(0)>

```

## Text token embedding

The text token embedding builds indices for text tokens. Such indexed tokens can
be used by instances of [`TokenEmbedding`](#mxnet.contrib.text.embedding.TokenEmbedding)
and [`Glossary`](#mxnet.contrib.text.glossary.Glossary).

To load token embeddings from an externally hosted pre-trained token embedding
file, such as those of GloVe and FastText, use
[`TokenEmbedding.create(embedding_name, pretrained_file_name)`](#mxnet.contrib.text.embedding.TokenEmbedding.create).
To get all the available `embedding_name` and `pretrained_file_name`, use
[`TokenEmbedding.get_embedding_and_pretrained_file_names()`](#mxnet.contrib.text.embedding.TokenEmbedding.get_embedding_and_pretrained_file_names).

Alternatively, to load embedding vectors from a custom pre-trained text token
embedding file, use [`CustomEmbedding`](#mxnet.contrib.text.embedding.CustomEmbedding).


```eval_rst
.. currentmodule:: mxnet.contrib.text.embedding
.. autosummary::
    :nosignatures:

    TokenEmbedding
    GloVe
    FastText
    CustomEmbedding
```

To get all the valid names for pre-trained embeddings and files, we can use
[`TokenEmbedding.get_embedding_and_pretrained_file_names`](#mxnet.contrib.text.embedding.TokenEmbedding.get_embedding_and_pretrained_file_names).

```python
>>> text.embedding.TokenEmbedding.get_embedding_and_pretrained_file_names()
{'glove': ['glove.42B.300d.txt', 'glove.6B.50d.txt', 'glove.6B.100d.txt',
'glove.6B.200d.txt', 'glove.6B.300d.txt', 'glove.840B.300d.txt',
'glove.twitter.27B.25d.txt', 'glove.twitter.27B.50d.txt',
'glove.twitter.27B.100d.txt', 'glove.twitter.27B.200d.txt'],
'fasttext': ['wiki.en.vec', 'wiki.simple.vec', 'wiki.zh.vec']}

```

To begin with, we can create a GloVe word embedding object by specifying the
embedding name `glove` and the pre-trained file `glove.6B.50d.txt`. The
argument `init_unknown_vec` specifies default vector representation for any
unknown token.

```python
>>> glove_6b_50d = text.embedding.TokenEmbedding.create('glove',
...     pretrained_file_name='glove.6B.50d.txt', init_unknown_vec=nd.zeros)

```

We can access properties such as `token_to_idx` (mapping tokens to indices),
`idx_to_token` (mapping indices to tokens), `vec_len` (length of each embedding
vector), and `unknown_token` (representation of any unknown token, default
value is '<unk>').

```python
>>> glove_6b_50d.token_to_idx['hi']
11084
>>> glove_6b_50d.idx_to_token[11084]
'hi'
>>> glove_6b_50d.vec_len
50
>>> glove_6b_50d.unknown_token
'<unk>'

```

For every unknown token, if its representation '<unk>' is encountered in the
pre-trained token embedding file, index 0 of property `idx_to_vec` maps to the
pre-trained token embedding vector loaded from the file; otherwise, index 0 of
property `idx_to_vec` maps to the default token embedding vector specified via
`init_unknown_vec` (set to nd.zeros here). Since the pre-trained file
does not have a vector for the token '<unk>', index 0 has to map to an
additional token '<unk>' and the number of tokens in the embedding is 400,001.


```python
>>> len(glove_6b_50d)
400001
>>> glove_6b_50d.idx_to_vec[0]

[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  ...
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
<NDArray 50 @cpu(0)>
>>> glove_6b_50d.get_vecs_by_tokens('unknownT0kEN')

[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  ...
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
<NDArray 50 @cpu(0)>
>>> glove_6b_50d.get_vecs_by_tokens(['unknownT0kEN', 'unknownT0kEN'])

[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
   ...
   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
   ...
   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
<NDArray 2x50 @cpu(0)>

```


### Implement a new text token embedding

For ``optimizer``, create a subclass of
[`TokenEmbedding`](#mxnet.contrib.text.embedding.TokenEmbedding).
Also add ``@TokenEmbedding.register`` before this class. See
[`embedding.py`](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/contrib/text/embedding.py)
for examples.


## Text token indexer

The text token indexer builds indices for text tokens. Such indexed tokens can
be used by instances of [`TokenEmbedding`](#mxnet.contrib.text.embedding.TokenEmbedding)
and [`Glossary`](#mxnet.contrib.text.glossary.Glossary). The input
counter whose keys are candidate indices may be obtained via
[`count_tokens_from_str`](#mxnet.contrib.text.utils.count_tokens_from_str).


```eval_rst
.. currentmodule:: mxnet.contrib.text.indexer
.. autosummary::
    :nosignatures:

    TokenIndexer
```

Suppose that we have a simple text data set in the string format. We can count
word frequency in the data set.

```python
>>> text_data = " hello world \n hello nice world \n hi world \n"
>>> counter = text.utils.count_tokens_from_str(text_data)

```

The obtained `counter` has key-value pairs whose keys are words and values are
word frequencies. Suppose that we want to build indices for the 2 most frequent
keys in `counter` with the unknown token representation '<UnK>' and a reserved
token '<pad>'.

```python
>>> token_indexer = text.indexer.TokenIndexer(counter, most_freq_count=2,
...     unknown_token='<UnK>', reserved_tokens=['<pad>'])

```

We can access properties such as `token_to_idx` (mapping tokens to indices),
`idx_to_token` (mapping indices to tokens), `vec_len` (length of each embedding
vector), and `unknown_token` (representation of any unknown token) and
`reserved_tokens`.

```python
>>> token_indexer = text.indexer.TokenIndexer(counter, most_freq_count=2,
...     unknown_token='<UnK>', reserved_tokens=['<pad>'])

```

```python
>>> token_indexer.token_to_idx
{'<UnK>': 0, '<pad>': 1, 'world': 2, 'hello': 3}
>>> token_indexer.idx_to_token
['<UnK>', '<pad>', 'world', 'hello']
>>> token_indexer.unknown_token
'<UnK>'
>>> token_indexer.reserved_tokens
['<pad>']
>>> len(token_indexer)
4
```

Besides the specified unknown token '<UnK>' and reserved_token '<pad>' are
indexed, the 2 most frequent words 'world' and 'hello' are also indexed.



## Text utilities

The following functions provide utilities for text data processing.

```eval_rst
.. currentmodule:: mxnet.contrib.text.utils
.. autosummary::
    :nosignatures:

    count_tokens_from_str
```




## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst

.. automodule:: mxnet.contrib.text.glossary
.. autoclass:: mxnet.contrib.text.glossary.Glossary
    :members: get_vecs_by_tokens, update_token_vectors, to_indices, to_tokens

.. automodule:: mxnet.contrib.text.embedding
.. autoclass:: mxnet.contrib.text.embedding.TokenEmbedding
    :members: get_vecs_by_tokens, update_token_vectors, to_indices, to_tokens, register, create, get_embedding_and_pretrained_file_names
.. autoclass:: mxnet.contrib.text.embedding.GloVe
    :members: get_vecs_by_tokens, update_token_vectors, to_indices, to_tokens
.. autoclass:: mxnet.contrib.text.embedding.FastText
    :members: get_vecs_by_tokens, update_token_vectors, to_indices, to_tokens
.. autoclass:: mxnet.contrib.text.embedding.CustomEmbedding
    :members: get_vecs_by_tokens, update_token_vectors, to_indices, to_tokens

.. automodule:: mxnet.contrib.text.indexer
.. autoclass:: mxnet.contrib.text.indexer.TokenIndexer
    :members: to_indices, to_tokens

.. automodule:: mxnet.contrib.text.utils
    :members: count_tokens_from_str

```
<script>auto_index("api-reference");</script>