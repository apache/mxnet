# Text API

## Overview

The mxnet.text APIs refer to classes and functions related to text data
processing, such as bulding indices and loading pre-trained embedding vectors
for text tokens and storing them in the `mxnet.ndarray` format.

This document lists the text APIs in mxnet:

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.text.glossary
    mxnet.text.embedding
    mxnet.text.indexer
    mxnet.text.utils
```


## Glossary

The glossary provides indexing and embedding for text tokens in a glossary. For
each indexed token in a glossary, an embedding vector will be associated with
it. Such embedding vectors can be loaded from externally hosted or custom
pre-trained token embedding files, such as via instances of
`mxnet.text.embedding.TokenEmbedding`. The input counter whose keys are candidate
indices may be obtained via `mxnet.text.utils.count_tokens_from_str`.

```eval_rst
.. currentmodule:: mxnet.text.glossary
.. autosummary::
    :nosignatures:

    Glossary
```

```python
>>> from mxnet.text.embedding import TokenEmbedding
>>> from mxnet.text.glossary import Glossary
>>> from collections import Counter
>>> fasttext_simple = TokenEmbedding.create('fasttext', 
...     pretrained_file_name='wiki.simple.vec')
>>> counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])
>>> gls = Glossary(counter, fasttext_simple, most_freq_count=None, min_freq=1,
...                unknown_token='<unk>', reserved_tokens=['<pad>'])
>>> 
>>> gls.token_to_idx
{'<unk>': 0, '<pad>': 1, 'c': 2, 'b': 3, 'a': 4, 'some_word$': 5}
>>> gls.idx_to_token
['<unk>', '<pad>', 'c', 'b', 'a', 'some_word$']
>>> len(gls)
6
>>> gls.vec_len
300
>>> gls.get_vecs_by_tokens('$unknownT0ken')

[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  ...
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
<NDArray 300 @cpu(0)>
```

## Text token embedding

The text token embedding builds indices for text tokens. Such indexed tokens can
be used by instances of `mxnet.text.embedding.TokenEmbedding` and
`mxnet.text.glossary.Glossary`.

To load token embeddings from an externally hosted pre-trained token embedding
file, such as those of GloVe and FastText, use
`TokenEmbedding.create(embedding_name, pretrained_file_name)`. To get all the
available `embedding_name` and `pretrained_file_name`, use
`TokenEmbedding.get_embedding_and_pretrained_file_names()`.

Alternatively, to load embedding vectors from a custom pre-trained text token
embedding file, use `mxnet.text.embeddings.CustomEmbedding`.


```eval_rst
.. currentmodule:: mxnet.text.embedding
.. autosummary::
    :nosignatures:

    TokenEmbedding
    GloVe
    FastText
    CustomEmbedding
```

```python
>>> from mxnet.text.embedding import TokenEmbedding
>>> TokenEmbedding.get_embedding_and_pretrained_file_names()
{'glove': ['glove.42B.300d.txt', 'glove.6B.50d.txt', 'glove.6B.100d.txt',
'glove.6B.200d.txt', 'glove.6B.300d.txt', 'glove.840B.300d.txt',
'glove.twitter.27B.25d.txt', 'glove.twitter.27B.50d.txt',
'glove.twitter.27B.100d.txt', 'glove.twitter.27B.200d.txt'],
'fasttext': ['wiki.en.vec', 'wiki.simple.vec', 'wiki.zh.vec']}
>>> glove_6b_50d = TokenEmbedding.create('glove',
...                                      pretrained_file_name='glove.6B.50d.txt')
>>> len(glove_6b_50d)
400001
>>> glove_6b_50d.vec_len
50
>>> glove_6b_50d.token_to_idx['hi']
11084
>>> glove_6b_50d.idx_to_token[11084]
'hi'
>>> # 0 is the index for any unknown token.
... glove_6b_50d.idx_to_vec[0]

[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  ...
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
<NDArray 50 @cpu(0)>
>>> glove_6b_50d.get_vecs_by_tokens('<unk$unk@unk>')

[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  ...
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
<NDArray 50 @cpu(0)>
>>> glove_6b_50d.get_vecs_by_tokens(['<unk$unk@unk>', '<unk$unk@unk>'])

[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
   ...
   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
   ...
   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
<NDArray 2x50 @cpu(0)>

```


## Implement a new text token embedding

For ``optimizer``, create a subclass of `mxnet.text.embedding.TokenEmbedding`.
Also add ``@TokenEmbedding.register`` before this class. See
[`embedding.py`](https://github.com/dmlc/mxnet/blob/master/python/mxnet/text/embedding.py)
for examples.


## Text token indexer

The text token indexer builds indices for text tokens. Such indexed tokens can
be used by instances of `mxnet.text.embedding.TokenEmbedding` and
`mxnet.text.glossary.Glossary`. The input counter whose keys are candidate
indices may be obtained via `mxnet.text.utils.count_tokens_from_str`.


```eval_rst
.. currentmodule:: mxnet.text.indexer
.. autosummary::
    :nosignatures:

    TokenIndexer
```

```python
>>> from mxnet.text.indexer import TokenIndexer
>>> from collections import Counter
>>> counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])
>>> token_indexer = TokenIndexer(counter, most_freq_count=None, min_freq=1,
...                              unknown_token='<unk>', 
...                              reserved_tokens=['<pad>'])
>>> len(token_indexer)
6
>>> token_indexer.token_to_idx
{'<unk>': 0, '<pad>': 1, 'c': 2, 'b': 3, 'a': 4, 'some_word$': 5}
>>> token_indexer.idx_to_token
['<unk>', '<pad>', 'c', 'b', 'a', 'some_word$']
>>> token_indexer.unknown_token
'<unk>'
>>> token_indexer.reserved_tokens
['<pad>']
>>> token_indexer2 = TokenIndexer(counter, most_freq_count=2, min_freq=3,
...                               unknown_token='<unk>', reserved_tokens=None)
>>> len(token_indexer2)
2
>>> token_indexer2.token_to_idx
{'<unk>': 0, 'c': 1}
>>> token_indexer2.idx_to_token
['<unk>', 'c']
>>> token_indexer2.unknown_token
'<unk>'
```



## Text utilities

The following functions provide utilities for text data processing.

```eval_rst
.. currentmodule:: mxnet.text.utils
.. autosummary::
    :nosignatures:

    count_tokens_from_str
    tokens_to_indices
    indices_to_tokens
```




## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst

.. automodule:: mxnet.text.glossary
.. autoclass:: mxnet.text.glossary.Glossary
    :members: get_vecs_by_tokens, update_token_vectors

.. automodule:: mxnet.text.embedding
.. autoclass:: mxnet.text.embedding.TokenEmbedding
    :members: get_vecs_by_tokens, update_token_vectors, register, create, get_embedding_and_pretrained_file_names
.. autoclass:: mxnet.text.embedding.GloVe
    :members: get_vecs_by_tokens, update_token_vectors
.. autoclass:: mxnet.text.embedding.FastText
    :members: get_vecs_by_tokens, update_token_vectors
.. autoclass:: mxnet.text.embedding.CustomEmbedding
    :members: get_vecs_by_tokens, update_token_vectors 

.. automodule:: mxnet.text.indexer
.. autoclass:: mxnet.text.indexer.TokenIndexer
    :members:

.. automodule:: mxnet.text.utils
    :members: count_tokens_from_str, tokens_to_indices, indices_to_tokens

```
<script>auto_index("api-reference");</script>
