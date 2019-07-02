# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=consider-iterating-dictionary
# pylint: disable=super-init-not-called

"""Text token embeddings."""
from __future__ import absolute_import
from __future__ import print_function

import io
import logging
import os
import tarfile
import warnings
import zipfile

from . import _constants as C
from . import vocab
from ... import ndarray as nd
from ... import registry
from ... import base
from ...util import is_np_array
from ... import numpy as _mx_np
from ... import numpy_extension as _mx_npx


def register(embedding_cls):
    """Registers a new token embedding.


    Once an embedding is registered, we can create an instance of this embedding with
    :func:`~mxnet.contrib.text.embedding.create`.


    Examples
    --------
    >>> @mxnet.contrib.text.embedding.register
    ... class MyTextEmbed(mxnet.contrib.text.embedding._TokenEmbedding):
    ...     def __init__(self, pretrained_file_name='my_pretrain_file'):
    ...         pass
    >>> embed = mxnet.contrib.text.embedding.create('MyTokenEmbed')
    >>> print(type(embed))
    <class '__main__.MyTokenEmbed'>
    """

    register_text_embedding = registry.get_register_func(_TokenEmbedding, 'token embedding')
    return register_text_embedding(embedding_cls)


def create(embedding_name, **kwargs):
    """Creates an instance of token embedding.


    Creates a token embedding instance by loading embedding vectors from an externally hosted
    pre-trained token embedding file, such as those of GloVe and FastText. To get all the valid
    `embedding_name` and `pretrained_file_name`, use
    `mxnet.contrib.text.embedding.get_pretrained_file_names()`.


    Parameters
    ----------
    embedding_name : str
        The token embedding name (case-insensitive).


    Returns
    -------
    An instance of `mxnet.contrib.text.glossary._TokenEmbedding`:
        A token embedding instance that loads embedding vectors from an externally hosted
        pre-trained token embedding file.
    """

    create_text_embedding = registry.get_create_func(_TokenEmbedding, 'token embedding')
    return create_text_embedding(embedding_name, **kwargs)


def get_pretrained_file_names(embedding_name=None):
    """Get valid token embedding names and their pre-trained file names.


    To load token embedding vectors from an externally hosted pre-trained token embedding file,
    such as those of GloVe and FastText, one should use
    `mxnet.contrib.text.embedding.create(embedding_name, pretrained_file_name)`.
    This method returns all the valid names of `pretrained_file_name` for the specified
    `embedding_name`. If `embedding_name` is set to None, this method returns all the valid
    names of `embedding_name` with their associated `pretrained_file_name`.


    Parameters
    ----------
    embedding_name : str or None, default None
        The pre-trained token embedding name.


    Returns
    -------
    dict or list:
        A list of all the valid pre-trained token embedding file names (`pretrained_file_name`)
        for the specified token embedding name (`embedding_name`). If the text embeding name is
        set to None, returns a dict mapping each valid token embedding name to a list of valid
        pre-trained files (`pretrained_file_name`). They can be plugged into
        `mxnet.contrib.text.embedding.create(embedding_name,
        pretrained_file_name)`.
    """

    text_embedding_reg = registry.get_registry(_TokenEmbedding)

    if embedding_name is not None:
        if embedding_name not in text_embedding_reg:
            raise KeyError('Cannot find `embedding_name` %s. Use '
                           '`get_pretrained_file_names('
                           'embedding_name=None).keys()` to get all the valid embedding '
                           'names.' % embedding_name)
        return list(text_embedding_reg[embedding_name].pretrained_file_name_sha1.keys())
    else:
        return {embedding_name: list(embedding_cls.pretrained_file_name_sha1.keys())
                for embedding_name, embedding_cls in registry.get_registry(_TokenEmbedding).items()}


class _TokenEmbedding(vocab.Vocabulary):
    """Token embedding base class.


    To load token embeddings from an externally hosted pre-trained token embedding file, such as
    those of GloVe and FastText, use
    :func:`~mxnet.contrib.text.embedding.create(embedding_name, pretrained_file_name)`.
    To get all the available `embedding_name` and `pretrained_file_name`, use
    :func:`~mxnet.contrib.text.embedding.get_pretrained_file_names()`.

    Alternatively, to load embedding vectors from a custom pre-trained token embedding file, use
    :class:`~mxnet.contrib.text.embedding.CustomEmbedding`.

    Moreover, to load composite embedding vectors, such as to concatenate embedding vectors, use
    :class:`~mxnet.contrib.text.embedding.CompositeEmbedding`.

    For every unknown token, if its representation `self.unknown_token` is encountered in the
    pre-trained token embedding file, index 0 of `self.idx_to_vec` maps to the pre-trained token
    embedding vector loaded from the file; otherwise, index 0 of `self.idx_to_vec` maps to the
    token embedding vector initialized by `init_unknown_vec`.

    If a token is encountered multiple times in the pre-trained token embedding file, only the
    first-encountered token embedding vector will be loaded and the rest will be skipped.

    The indexed tokens in a text token embedding may come from a vocabulary or from the loaded
    embedding vectors. In the former case, only the indexed tokens in a vocabulary are associated
    with the loaded embedding vectors, such as loaded from a pre-trained token embedding file. In
    the later case, all the tokens from the loaded embedding vectors, such as loaded from a
    pre-trained token embedding file, are taken as the indexed tokens of the embedding.


    Attributes
    ----------
    token_to_idx : dict mapping str to int
        A dict mapping each token to its index integer.
    idx_to_token : list of strs
        A list of indexed tokens where the list indices and the token indices are aligned.
    unknown_token : hashable object
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation.
    reserved_tokens : list of strs or None
        A list of reserved tokens that will always be indexed.
    vec_len : int
        The length of the embedding vector for each token.
    idx_to_vec : mxnet.ndarray.NDArray
        For all the indexed tokens in this embedding, this NDArray maps each token's index to an
        embedding vector. The largest valid index maps to the initialized embedding vector for every
        reserved token, such as an unknown_token token and a padding token.
    """

    def __init__(self, **kwargs):
        super(_TokenEmbedding, self).__init__(**kwargs)

    @classmethod
    def _get_download_file_name(cls, pretrained_file_name):
        return pretrained_file_name

    @classmethod
    def _get_pretrained_file_url(cls, pretrained_file_name):
        repo_url = os.environ.get('MXNET_GLUON_REPO', C.APACHE_REPO_URL)
        embedding_cls = cls.__name__.lower()

        url_format = '{repo_url}gluon/embeddings/{cls}/{file_name}'
        return url_format.format(repo_url=repo_url, cls=embedding_cls,
                                 file_name=cls._get_download_file_name(pretrained_file_name))

    @classmethod
    def _get_pretrained_file(cls, embedding_root, pretrained_file_name):
        from ...gluon.utils import check_sha1, download
        embedding_cls = cls.__name__.lower()
        embedding_root = os.path.expanduser(embedding_root)
        url = cls._get_pretrained_file_url(pretrained_file_name)

        embedding_dir = os.path.join(embedding_root, embedding_cls)
        pretrained_file_path = os.path.join(embedding_dir, pretrained_file_name)
        downloaded_file = os.path.basename(url)
        downloaded_file_path = os.path.join(embedding_dir, downloaded_file)

        expected_file_hash = cls.pretrained_file_name_sha1[pretrained_file_name]

        if hasattr(cls, 'pretrained_archive_name_sha1'):
            expected_downloaded_hash = \
                cls.pretrained_archive_name_sha1[downloaded_file]
        else:
            expected_downloaded_hash = expected_file_hash

        if not os.path.exists(pretrained_file_path) \
           or not check_sha1(pretrained_file_path, expected_file_hash):
            download(url, downloaded_file_path, sha1_hash=expected_downloaded_hash)

            ext = os.path.splitext(downloaded_file)[1]
            if ext == '.zip':
                with zipfile.ZipFile(downloaded_file_path, 'r') as zf:
                    zf.extractall(embedding_dir)
            elif ext == '.gz':
                with tarfile.open(downloaded_file_path, 'r:gz') as tar:
                    tar.extractall(path=embedding_dir)
        return pretrained_file_path

    def _load_embedding(self, pretrained_file_path, elem_delim, init_unknown_vec, encoding='utf8'):
        """Load embedding vectors from the pre-trained token embedding file.


        For every unknown token, if its representation `self.unknown_token` is encountered in the
        pre-trained token embedding file, index 0 of `self.idx_to_vec` maps to the pre-trained token
        embedding vector loaded from the file; otherwise, index 0 of `self.idx_to_vec` maps to the
        text embedding vector initialized by `init_unknown_vec`.

        If a token is encountered multiple times in the pre-trained text embedding file, only the
        first-encountered token embedding vector will be loaded and the rest will be skipped.
        """

        pretrained_file_path = os.path.expanduser(pretrained_file_path)

        if not os.path.isfile(pretrained_file_path):
            raise ValueError('`pretrained_file_path` must be a valid path to '
                             'the pre-trained token embedding file.')

        logging.info('Loading pre-trained token embedding vectors from %s', pretrained_file_path)
        vec_len = None
        all_elems = []
        tokens = set()
        loaded_unknown_vec = None
        line_num = 0
        with io.open(pretrained_file_path, 'r', encoding=encoding) as f:
            for line in f:
                line_num += 1
                elems = line.rstrip().split(elem_delim)

                assert len(elems) > 1, 'At line %d of the pre-trained text embedding file: the ' \
                                       'data format of the pre-trained token embedding file %s ' \
                                       'is unexpected.' % (line_num, pretrained_file_path)

                token, elems = elems[0], [float(i) for i in elems[1:]]

                if token == self.unknown_token and loaded_unknown_vec is None:
                    loaded_unknown_vec = elems
                    tokens.add(self.unknown_token)
                elif token in tokens:
                    warnings.warn('At line %d of the pre-trained token embedding file: the '
                                  'embedding vector for token %s has been loaded and a duplicate '
                                  'embedding for the  same token is seen and skipped.' %
                                  (line_num, token))
                elif len(elems) == 1:
                    warnings.warn('At line %d of the pre-trained text embedding file: token %s '
                                  'with 1-dimensional vector %s is likely a header and is '
                                  'skipped.' % (line_num, token, elems))
                else:
                    if vec_len is None:
                        vec_len = len(elems)
                        # Reserve a vector slot for the unknown token at the very beggining because
                        # the unknown index is 0.
                        all_elems.extend([0] * vec_len)
                    else:
                        assert len(elems) == vec_len, \
                            'At line %d of the pre-trained token embedding file: the dimension ' \
                            'of token %s is %d but the dimension of previous tokens is %d. ' \
                            'Dimensions of all the tokens must be the same.' \
                            % (line_num, token, len(elems), vec_len)
                    all_elems.extend(elems)
                    self._idx_to_token.append(token)
                    self._token_to_idx[token] = len(self._idx_to_token) - 1
                    tokens.add(token)

        self._vec_len = vec_len
        array_fn = _mx_np.array if is_np_array() else nd.array
        self._idx_to_vec = array_fn(all_elems).reshape((-1, self.vec_len))

        if loaded_unknown_vec is None:
            init_val = init_unknown_vec(shape=self.vec_len)
            self._idx_to_vec[C.UNKNOWN_IDX] =\
                init_val.as_np_ndarray() if is_np_array() else init_val
        else:
            self._idx_to_vec[C.UNKNOWN_IDX] = array_fn(loaded_unknown_vec)

    def _index_tokens_from_vocabulary(self, vocabulary):
        self._token_to_idx = vocabulary.token_to_idx.copy() \
            if vocabulary.token_to_idx is not None else None
        self._idx_to_token = vocabulary.idx_to_token[:] \
            if vocabulary.idx_to_token is not None else None
        self._unknown_token = vocabulary.unknown_token
        self._reserved_tokens = vocabulary.reserved_tokens[:] \
            if vocabulary.reserved_tokens is not None else None

    def _set_idx_to_vec_by_embeddings(self, token_embeddings, vocab_len, vocab_idx_to_token):
        """Sets the mapping between token indices and token embedding vectors.


        Parameters
        ----------
        token_embeddings : instance or list `mxnet.contrib.text.embedding._TokenEmbedding`
            One or multiple pre-trained token embeddings to load. If it is a list of multiple
            embeddings, these embedding vectors will be concatenated for each token.
        vocab_len : int
            Length of vocabulary whose tokens are indexed in the token embedding.
        vocab_idx_to_token: list of str
            A list of indexed tokens in the vocabulary. These tokens are indexed in the token
            embedding.
        """

        new_vec_len = sum(embed.vec_len for embed in token_embeddings)
        zeros_fn = _mx_np.zeros if is_np_array() else nd.zeros
        new_idx_to_vec = zeros_fn(shape=(vocab_len, new_vec_len))

        col_start = 0
        # Concatenate all the embedding vectors in token_embeddings.
        for embed in token_embeddings:
            col_end = col_start + embed.vec_len
            # Cancatenate vectors of the unknown token.
            new_idx_to_vec[0, col_start:col_end] = embed.idx_to_vec[0]
            new_idx_to_vec[1:, col_start:col_end] = embed.get_vecs_by_tokens(vocab_idx_to_token[1:])
            col_start = col_end

        self._vec_len = new_vec_len
        self._idx_to_vec = new_idx_to_vec

    def _build_embedding_for_vocabulary(self, vocabulary):
        if vocabulary is not None:
            assert isinstance(vocabulary, vocab.Vocabulary), \
                'The argument `vocabulary` must be an instance of ' \
                'mxnet.contrib.text.vocab.Vocabulary.'

            # Set _idx_to_vec so that indices of tokens from vocabulary are associated with the
            # loaded token embedding vectors.
            self._set_idx_to_vec_by_embeddings([self], len(vocabulary), vocabulary.idx_to_token)

            # Index tokens from vocabulary.
            self._index_tokens_from_vocabulary(vocabulary)

    @property
    def vec_len(self):
        return self._vec_len

    @property
    def idx_to_vec(self):
        return self._idx_to_vec

    def get_vecs_by_tokens(self, tokens, lower_case_backup=False):
        """Look up embedding vectors of tokens.


        Parameters
        ----------
        tokens : str or list of strs
            A token or a list of tokens.
        lower_case_backup : bool, default False
            If False, each token in the original case will be looked up; if True, each token in the
            original case will be looked up first, if not found in the keys of the property
            `token_to_idx`, the token in the lower case will be looked up.


        Returns
        -------
        mxnet.ndarray.NDArray:
            The embedding vector(s) of the token(s). According to numpy conventions, if `tokens` is
            a string, returns a 1-D NDArray of shape `self.vec_len`; if `tokens` is a list of
            strings, returns a 2-D NDArray of shape=(len(tokens), self.vec_len).
        """

        to_reduce = False
        if not isinstance(tokens, list):
            tokens = [tokens]
            to_reduce = True

        if not lower_case_backup:
            indices = [self.token_to_idx.get(token, C.UNKNOWN_IDX) for token in tokens]
        else:
            indices = [self.token_to_idx[token] if token in self.token_to_idx
                       else self.token_to_idx.get(token.lower(), C.UNKNOWN_IDX)
                       for token in tokens]

        if is_np_array():
            embedding_fn = _mx_npx.embedding
            array_fn = _mx_np.array
        else:
            embedding_fn = nd.Embedding
            array_fn = nd.array
        vecs = embedding_fn(array_fn(indices), self.idx_to_vec, self.idx_to_vec.shape[0],
                            self.idx_to_vec.shape[1])

        return vecs[0] if to_reduce else vecs

    def update_token_vectors(self, tokens, new_vectors):
        """Updates embedding vectors for tokens.


        Parameters
        ----------
        tokens : str or a list of strs
            A token or a list of tokens whose embedding vector are to be updated.
        new_vectors : mxnet.ndarray.NDArray
            An NDArray to be assigned to the embedding vectors of `tokens`. Its length must be equal
            to the number of `tokens` and its width must be equal to the dimension of embeddings of
            the glossary. If `tokens` is a singleton, it must be 1-D or 2-D. If `tokens` is a list
            of multiple strings, it must be 2-D.
        """

        assert self.idx_to_vec is not None, 'The property `idx_to_vec` has not been properly set.'

        if not isinstance(tokens, list) or len(tokens) == 1:
            assert isinstance(new_vectors, nd.NDArray) and len(new_vectors.shape) in [1, 2], \
                '`new_vectors` must be a 1-D or 2-D NDArray if `tokens` is a singleton.'
            if not isinstance(tokens, list):
                tokens = [tokens]
            if len(new_vectors.shape) == 1:
                expand_dims_fn = _mx_np.expand_dims if is_np_array() else nd.expand_dims
                new_vectors = expand_dims_fn(new_vectors, axis=0)

        else:
            assert isinstance(new_vectors, nd.NDArray) and len(new_vectors.shape) == 2, \
                '`new_vectors` must be a 2-D NDArray if `tokens` is a list of multiple strings.'
        assert new_vectors.shape == (len(tokens), self.vec_len), \
            'The length of new_vectors must be equal to the number of tokens and the width of' \
            'new_vectors must be equal to the dimension of embeddings of the glossary.'

        indices = []
        for token in tokens:
            if token in self.token_to_idx:
                indices.append(self.token_to_idx[token])
            else:
                raise ValueError('Token %s is unknown. To update the embedding vector for an '
                                 'unknown token, please specify it explicitly as the '
                                 '`unknown_token` %s in `tokens`. This is to avoid unintended '
                                 'updates.' % (token, self.idx_to_token[C.UNKNOWN_IDX]))

        array_fn = _mx_np.array if is_np_array() else nd.array
        self._idx_to_vec[array_fn(indices)] = new_vectors

    @classmethod
    def _check_pretrained_file_names(cls, pretrained_file_name):
        """Checks if a pre-trained token embedding file name is valid.


        Parameters
        ----------
        pretrained_file_name : str
            The pre-trained token embedding file.
        """

        embedding_name = cls.__name__.lower()
        if pretrained_file_name not in cls.pretrained_file_name_sha1:
            raise KeyError('Cannot find pretrained file %s for token embedding %s. Valid '
                           'pretrained files for embedding %s: %s' %
                           (pretrained_file_name, embedding_name, embedding_name,
                            ', '.join(cls.pretrained_file_name_sha1.keys())))


@register
class GloVe(_TokenEmbedding):
    """The GloVe word embedding.


    GloVe is an unsupervised learning algorithm for obtaining vector representations for words.
    Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and
    the resulting representations showcase interesting linear substructures of the word vector
    space. (Source from https://nlp.stanford.edu/projects/glove/)

    References
    ----------

    GloVe: Global Vectors for Word Representation.
    Jeffrey Pennington, Richard Socher, and Christopher D. Manning.
    https://nlp.stanford.edu/pubs/glove.pdf

    Website:

    https://nlp.stanford.edu/projects/glove/

    To get the updated URLs to the externally hosted pre-trained token embedding
    files, visit https://nlp.stanford.edu/projects/glove/

    License for pre-trained embeddings:

        https://fedoraproject.org/wiki/Licensing/PDDL


    Parameters
    ----------
    pretrained_file_name : str, default 'glove.840B.300d.txt'
        The name of the pre-trained token embedding file.
    embedding_root : str, default $MXNET_HOME/embeddings
        The root directory for storing embedding-related files.
    init_unknown_vec : callback
        The callback used to initialize the embedding vector for the unknown token.
    vocabulary : :class:`~mxnet.contrib.text.vocab.Vocabulary`, default None
        It contains the tokens to index. Each indexed token will be associated with the loaded
        embedding vectors, such as loaded from a pre-trained token embedding file. If None, all the
        tokens from the loaded embedding vectors, such as loaded from a pre-trained token embedding
        file, will be indexed.
    """

    # Map a pre-trained token embedding archive file and its SHA-1 hash.
    pretrained_archive_name_sha1 = C.GLOVE_PRETRAINED_FILE_SHA1

    # Map a pre-trained token embedding file and its SHA-1 hash.
    pretrained_file_name_sha1 = C.GLOVE_PRETRAINED_ARCHIVE_SHA1

    @classmethod
    def _get_download_file_name(cls, pretrained_file_name):
        # Map a pre-trained embedding file to its archive to download.
        src_archive = {archive.split('.')[1]: archive for archive in
                       GloVe.pretrained_archive_name_sha1.keys()}
        archive = src_archive[pretrained_file_name.split('.')[1]]
        return archive

    def __init__(self, pretrained_file_name='glove.840B.300d.txt',
                 embedding_root=os.path.join(base.data_dir(), 'embeddings'),
                 init_unknown_vec=nd.zeros, vocabulary=None, **kwargs):
        GloVe._check_pretrained_file_names(pretrained_file_name)

        super(GloVe, self).__init__(**kwargs)
        pretrained_file_path = GloVe._get_pretrained_file(embedding_root, pretrained_file_name)

        self._load_embedding(pretrained_file_path, ' ', init_unknown_vec)

        if vocabulary is not None:
            self._build_embedding_for_vocabulary(vocabulary)


@register
class FastText(_TokenEmbedding):
    """The fastText word embedding.


    FastText is an open-source, free, lightweight library that allows users to learn text
    representations and text classifiers. It works on standard, generic hardware. Models can later
    be reduced in size to even fit on mobile devices. (Source from https://fasttext.cc/)

    References
    ----------

    Enriching Word Vectors with Subword Information.
    Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov.
    https://arxiv.org/abs/1607.04606

    Bag of Tricks for Efficient Text Classification.
    Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov.
    https://arxiv.org/abs/1607.01759

    FastText.zip: Compressing text classification models.
    Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Herve Jegou,
    and Tomas Mikolov.
    https://arxiv.org/abs/1612.03651

    For 'wiki.multi' embeddings:
    Word Translation Without Parallel Data
    Alexis Conneau, Guillaume Lample, Marc'Aurelio Ranzato, Ludovic Denoyer,
    and Herve Jegou.
    https://arxiv.org/abs/1710.04087

    Website:

    https://fasttext.cc/

    To get the updated URLs to the externally hosted pre-trained token embedding files, visit
    https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md

    License for pre-trained embeddings:

        https://creativecommons.org/licenses/by-sa/3.0/


    Parameters
    ----------
    pretrained_file_name : str, default 'wiki.en.vec'
        The name of the pre-trained token embedding file.
    embedding_root : str, default $MXNET_HOME/embeddings
        The root directory for storing embedding-related files.
    init_unknown_vec : callback
        The callback used to initialize the embedding vector for the unknown token.
    vocabulary : :class:`~mxnet.contrib.text.vocab.Vocabulary`, default None
        It contains the tokens to index. Each indexed token will be associated with the loaded
        embedding vectors, such as loaded from a pre-trained token embedding file. If None, all the
        tokens from the loaded embedding vectors, such as loaded from a pre-trained token embedding
        file, will be indexed.
    """

    # Map a pre-trained token embedding archive file and its SHA-1 hash.
    pretrained_archive_name_sha1 = C.FAST_TEXT_ARCHIVE_SHA1

    # Map a pre-trained token embedding file and its SHA-1 hash.
    pretrained_file_name_sha1 = C.FAST_TEXT_FILE_SHA1

    @classmethod
    def _get_download_file_name(cls, pretrained_file_name):
        # Map a pre-trained embedding file to its archive to download.
        return '.'.join(pretrained_file_name.split('.')[:-1])+'.zip'

    def __init__(self, pretrained_file_name='wiki.simple.vec',
                 embedding_root=os.path.join(base.data_dir(), 'embeddings'),
                 init_unknown_vec=nd.zeros, vocabulary=None, **kwargs):
        FastText._check_pretrained_file_names(pretrained_file_name)

        super(FastText, self).__init__(**kwargs)
        pretrained_file_path = FastText._get_pretrained_file(embedding_root, pretrained_file_name)

        self._load_embedding(pretrained_file_path, ' ', init_unknown_vec)

        if vocabulary is not None:
            self._build_embedding_for_vocabulary(vocabulary)


class CustomEmbedding(_TokenEmbedding):
    """User-defined token embedding.

    This is to load embedding vectors from a user-defined pre-trained text embedding file.

    Denote by '[ed]' the argument `elem_delim`. Denote by [v_ij] the j-th element of the token
    embedding vector for [token_i], the expected format of a custom pre-trained token embedding file
    is:

    '[token_1][ed][v_11][ed][v_12][ed]...[ed][v_1k]\\\\n[token_2][ed][v_21][ed][v_22][ed]...[ed]
    [v_2k]\\\\n...'

    where k is the length of the embedding vector `vec_len`.


    Parameters
    ----------
    pretrained_file_path : str
        The path to the custom pre-trained token embedding file.
    elem_delim : str, default ' '
        The delimiter for splitting a token and every embedding vector element value on the same
        line of the custom pre-trained token embedding file.
    encoding : str, default 'utf8'
        The encoding scheme for reading the custom pre-trained token embedding file.
    init_unknown_vec : callback
        The callback used to initialize the embedding vector for the unknown token.
    vocabulary : :class:`~mxnet.contrib.text.vocab.Vocabulary`, default None
        It contains the tokens to index. Each indexed token will be associated with the loaded
        embedding vectors, such as loaded from a pre-trained token embedding file. If None, all the
        tokens from the loaded embedding vectors, such as loaded from a pre-trained token embedding
        file, will be indexed.
    """

    def __init__(self, pretrained_file_path, elem_delim=' ', encoding='utf8',
                 init_unknown_vec=nd.zeros, vocabulary=None, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self._load_embedding(pretrained_file_path, elem_delim, init_unknown_vec, encoding)

        if vocabulary is not None:
            self._build_embedding_for_vocabulary(vocabulary)


class CompositeEmbedding(_TokenEmbedding):
    """Composite token embeddings.


    For each indexed token in a vocabulary, multiple embedding vectors, such as concatenated
    multiple embedding vectors, will be associated with it. Such embedding vectors can be loaded
    from externally hosted or custom pre-trained token embedding files, such as via token embedding
    instances.


    Parameters
    ----------
    vocabulary : :class:`~mxnet.contrib.text.vocab.Vocabulary`
        For each indexed token in a vocabulary, multiple embedding vectors, such as concatenated
        multiple embedding vectors, will be associated with it.
    token_embeddings : instance or list of `mxnet.contrib.text.embedding._TokenEmbedding`
        One or multiple pre-trained token embeddings to load. If it is a list of multiple
        embeddings, these embedding vectors will be concatenated for each token.
    """
    def __init__(self, vocabulary, token_embeddings):

        # Sanity checks.
        assert isinstance(vocabulary, vocab.Vocabulary), \
            'The argument `vocabulary` must be an instance of ' \
            'mxnet.contrib.text.indexer.Vocabulary.'

        if not isinstance(token_embeddings, list):
            token_embeddings = [token_embeddings]

        for embed in token_embeddings:
            assert isinstance(embed, _TokenEmbedding), \
                'The argument `token_embeddings` must be an instance or a list of instances ' \
                'of `mxnet.contrib.text.embedding.TextEmbedding` whose embedding vectors will be' \
                'loaded or concatenated-then-loaded to map to the indexed tokens.'

        # Index tokens.
        self._index_tokens_from_vocabulary(vocabulary)

        # Set _idx_to_vec so that indices of tokens from keys of `counter` are associated with token
        # embedding vectors from `token_embeddings`.
        self._set_idx_to_vec_by_embeddings(token_embeddings, len(self), self.idx_to_token)
