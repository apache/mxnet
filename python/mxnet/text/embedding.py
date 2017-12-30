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
# pylint: disable=not-callable, invalid-encoded-data, dangerous-default-value
# pylint: disable=logging-not-lazy, consider-iterating-dictionary
# pylint: disable=raising-bad-type, super-init-not-called

"""Read text files and load embeddings."""
from __future__ import absolute_import
from __future__ import print_function

from collections import Counter
import io
import logging
import os
import tarfile
import warnings
import zipfile

from ..gluon.utils import check_sha1
from ..gluon.utils import download
from .. import ndarray as nd


class TextIndexer(object):
    """Indexing for text tokens.


    Build indices for the unknown token, reserved tokens, and input counter
    keys. Indexed tokens can be used by instances of
    :func:`~mxnet.text.embeddings.TextEmbed`.


    Parameters
    ----------
    counter : collections.Counter or None
        Counts text token frequencies in the text data.
    most_freq_count : None or int, default None
        The maximum possible number of the most frequent tokens in the keys of
        `counter` that can be indexed. Note that this argument does not count
        any token from `reserved_tokens`. If this argument is None or larger
        than its largest possible value restricted by `counter` and
        `reserved_tokens`, this argument becomes positive infinity.
    min_freq : int, default 1
        The minimum frequency required for a token in the keys of `counter` to
        be indexed.
    unknown_token : str, default '<unk>'
        The string representation for any unknown token. In other words, any
        unknown token will be indexed as the same string representation. This
        string representation cannot be any token to be indexed from the keys of
        `counter` or from `reserved_tokens`.
    reserved_tokens : list of strs or None, default None
        A list of reserved tokens that will always be indexed. It cannot contain
        `unknown_token`, or duplicate reserved tokens.


    Properties
    ----------
    token_to_idx : dict mapping str to int
        A dict mapping each token to its index integer.
    idx_to_token : list of strs
        A list of indexed tokens where the list indices and the token indices
        are aligned.
    unknown_token : str
        The string representation for any unknown token. In other words, any
        unknown token will be indexed as the same string representation.
    reserved_tokens : list of strs or None
        A list of reserved tokens that will always be indexed.
    unknown_idx : int
        The index for `unknown_token`.
    """
    def __init__(self, counter, most_freq_count=None, min_freq=1,
                 unknown_token='<unk>', reserved_tokens=None):
        # Sanity checks.
        assert min_freq > 0, '`min_freq` must be set to a positive value.'

        if reserved_tokens is not None:
            for reserved_token in reserved_tokens:
                assert reserved_token != unknown_token, \
                    '`reserved_token` cannot contain `unknown_token`.'
            assert len(set(reserved_tokens)) == len(reserved_tokens), \
                '`reserved_tokens` cannot contain duplicate reserved tokens.'

        self._index_unknown_and_reserved_tokens(unknown_token, reserved_tokens)

        if counter is not None:
            self._index_counter_keys(counter, unknown_token, reserved_tokens,
                                     most_freq_count, min_freq)

    def _index_unknown_and_reserved_tokens(self, unknown_token,
                                           reserved_tokens):
        """Indexes unknown and reserved tokens."""
        self._unknown_token = unknown_token
        self._idx_to_token = [unknown_token]

        if reserved_tokens is None:
            self._reserved_tokens = None
        else:
            # Python 2 does not support list.copy().
            self._reserved_tokens = reserved_tokens[:]
            self._idx_to_token.extend(reserved_tokens)

        self._token_to_idx = {token: idx for idx, token in
                              enumerate(self._idx_to_token)}

    def _index_counter_keys(self, counter, unknown_token, reserved_tokens,
                            most_freq_count, min_freq):
        """Indexes keys of `counter`.

        Indexes keys of `counter` according to frequency thresholds such as
        `most_freq_count` and `min_freq`.
        """
        assert isinstance(counter, Counter), \
            '`counter` must be an instance of collections.Counter.'

        if reserved_tokens is not None:
            reserved_tokens = set(reserved_tokens)
        else:
            reserved_tokens = set()

        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)

        if most_freq_count is None:
            # 1 is the unknown token count.
            token_cap = 1 + len(reserved_tokens) + len(counter)
        else:
            token_cap = 1 + len(reserved_tokens) + most_freq_count

        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) == token_cap:
                break
            if token == unknown_token:
                raise(ValueError, 'Keys of `counter` cannot contain '
                                  '`unknown_token`. Set `unknown_token` to '
                                  'another string representation.')
            elif token not in reserved_tokens:
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def unknown_token(self):
        return self._unknown_token

    @property
    def reserved_tokens(self):
        return self._reserved_tokens

    @property
    def unknown_idx(self):
        return 0


class TextEmbed(TextIndexer):
    """Text embedding.

    To load text embeddings from an externally hosted pre-trained file, such as
    pre-trained embedding files of GloVe and FastText, use
    `TextEmbed.create(embed_name, pretrain_file)`. To get all the
    available `embed_name` and `pretrain_file`, use
    TextEmbed.get_embed_names_and_pretrain_files().

    Alternatively, to load embeddings from a local pre-trained file, specify its
    local path via `pretrain_file` and set url to None. Denote by v_ij the j-th
    element of the text embedding vector for token_i, the expected format of a
    local pre-trained file is:

    token_1`token_delim`v_11`token_delim`v_12`token_delim`...`token_delim`v_1k\n
    token_2`token_delim`v_21`token_delim`v_22`token_delim`...`token_delim`v_2k\n
    ...

    where k is the length of the embedding vecgor `vec_len`.

    For the same token, its index and embedding vector may vary across different
    instances of mxnet.text.glossary.TextEmbed.


    Parameters
    ----------
    pretrain_file : str
        The name or path of the pre-trained embedding file. If the pre-trained
        embedding file is externally hosted at `url`, `pretrain_file` must be
        the name of the pre-trained embedding file; if the pre-trained embedding
        file is local, `pretrain_file` must be the path of the pre-trained
        embedding file and `url` must be set to None.
    url : str or None, default None
        The url where the pre-trained embedding file is externally hosted. If
        the pre-trained embedding file is externally hosted at `url`,
        `pretrain_file` must be the name of the pre-trained embedding file; if
        the pre-trained embedding file is local, `pretrain_file` must be the
        path of the pre-trained embedding file and `url` must be set to None.
    embed_name : str, default 'my_embed'
        The name space for embedding, such as 'glove' and 'fasttext'.
    embed_root : str, default '~/.mxnet/embeddings/'
        The root directory for storing embedding-related files.
    reserved_init_vec : callback, default mxnet.ndarray.zeros
        The callback used to initialize the embedding vector for every reserved
        token, such as an unknown_token token and a padding token.
    token_delim : str, default ' '
        The delimiter for splitting a token and every embedding vector element
        value on the same line of the pre-trained embedding file.


    Properties
    ----------
    vec_len : int
        The length of the embedding vector for each token.
    reserved_init_vec : callback
        The callback used to initialize the embedding vector for every reserved
        token.
    token_to_idx : dict mapping str to int
        A dict mapping each token to its index integer.
    idx_to_token : list of strs
        A list of indexed tokens where the list indices and the token indices
        are aligned.
    idx_to_vec : mxnet.ndarray.NDArray
        For all the indexed tokens in this embedding, this NDArray maps each
        token's index to an embedding vector. The largest valid index maps
        to the initialized embedding vector for every reserved token, such as an
        unknown_token token and a padding token.
    """

    # Key-value pairs for text embedding name in lower case and text embedding
    # class.
    embed_registry = {}

    def __init__(self, unknown_vec=nd.zeros, **kwargs):
        self._unknown_vec = unknown_vec
        super(TextEmbed, self).__init__(counter=None, reserved_tokens=None,
                                        **kwargs)

    @staticmethod
    def _get_pretrain_file_path_from_url(url, embed_root, embed_name,
                                         pretrain_file):
        """Get the local path to the pre-trained file from the url.

        The pretrained embedding file will be downloaded from url if it has not
        been downloaded yet or the existing file fails to match its expected
        SHA-1 hash.
        """
        embed_root = os.path.expanduser(embed_root)

        embed_dir = os.path.join(embed_root, embed_name)
        pretrain_file_path = os.path.join(embed_dir, pretrain_file)
        download_file = os.path.basename(url)
        download_file_path = os.path.join(embed_dir, download_file)

        embed_cls = TextEmbed.embed_registry[embed_name]
        expected_file_hash = embed_cls.pretrain_file_sha1[pretrain_file]

        if hasattr(embed_cls, 'pretrain_archive_sha1'):
            expected_download_hash = \
                embed_cls.pretrain_archive_sha1[download_file]
        else:
            expected_download_hash = expected_file_hash

        # The specified pretrained embedding file does not exist or fails to
        # match its expected SHA-1 hash.
        if not os.path.isfile(pretrain_file_path) or \
                not check_sha1(pretrain_file_path, expected_file_hash):
            # If download_file_path exists and matches
            # expected_download_hash, there is no need to download.
            download(url, download_file_path,
                     sha1_hash=expected_download_hash)

        # If the downloaded file does not match its expected SHA-1 hash,
        # we do not encourage to load embeddings from it in case that its
        # data format is changed.
        assert check_sha1(download_file_path, expected_download_hash), \
            'The downloaded file %s does not match its expected SHA-1 ' \
            'hash. This is caused by the changes at the externally ' \
            'hosted pretrained embedding file(s). Since its data format ' \
            'may also be changed, it is discouraged to continue to use ' \
            'mxnet.text.glossary.TextEmbed.create(%s, **kwargs) ' \
            'to load the pretrained embedding %s. If you still wish to load ' \
            'the changed embedding file, please specify its path %s via ' \
            'pretrain_file of mxnet.text.glossary.TextEmbed(). It will be ' \
            'loaded only if its data format remains unchanged.' % \
            (download_file_path, embed_name, embed_name, download_file_path)

        ext = os.path.splitext(download_file)[1]
        if ext == '.zip':
            with zipfile.ZipFile(download_file_path, 'r') as zf:
                zf.extractall(embed_dir)
        elif ext == '.gz':
            with tarfile.open(download_file_path, 'r:gz') as tar:
                tar.extractall(path=embed_dir)
        return pretrain_file_path

    def _load_embedding(self, pretrain_file_path, elem_delim):
        """Load embedding vectors from the pre-trained embedding file.

        The largest valid index of idx_to_vec maps to the initialized embedding
        vector for every reserved token, such as an unknown_token token and a padding
        token.
        """
        pretrain_file_path = os.path.expanduser(pretrain_file_path)

        if not os.path.isfile(pretrain_file_path):
            raise ValueError('`pretrain_file_path` must be a valid path to '
                             'the pre-trained text embedding file.')

        with io.open(pretrain_file_path, 'r', encoding='utf8') as f:
            lines = f.readlines()

        logging.info('Loading pretrained embedding vectors from %s'
                     % pretrain_file_path)

        vec_len = None
        all_elems = []
        tokens = set()
        for line in lines:
            elems = line.rstrip().split(elem_delim)

            assert len(elems) > 1, 'The data format of the pretrained ' \
                                   'embedding file %s is unexpected.' \
                                   % pretrain_file_path

            token, elems = elems[0], [float(i) for i in elems[1:]]

            if token in tokens:
                warnings.warn('The embedding vector for token %s has been '
                              'loaded and a duplicate embedding for the same '
                              'token is seen and skipped.' % token)
            else:
                if len(elems) == 1:
                    warnings.warn('Token %s with 1-dimensional vector %s is '
                                  'likely a header and is skipped.' %
                                  (token, elems))
                    continue
                else:
                    if vec_len is None:
                        vec_len = len(elems)
                        # Reserve a vector slot for the unknown token at the
                        # very beggining because the unknown index is 0.
                        all_elems.extend([0] * vec_len)
                    else:
                        assert len(elems) == vec_len, \
                            'The dimension of token %s is %d but the dimension ' \
                            'of previous tokens is %d. Dimensions of all the ' \
                            'tokens must be the same.' % (token, len(elems),
                                                          vec_len)
                all_elems.extend(elems)
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1
                tokens.add(token)

        self._vec_len = vec_len
        self._idx_to_vec = nd.array(all_elems).reshape((-1, self.vec_len))
        self._idx_to_vec[self.unknown_idx] = \
            self.unknown_vec(shape=self.vec_len)

    @property
    def unknown_vec(self):
        return self._unknown_vec

    @property
    def vec_len(self):
        return self._vec_len

    @property
    def idx_to_vec(self):
        return self._idx_to_vec

    def __getitem__(self, tokens):
        """The getter.

        Parameters
        ----------
        tokens : str or list of strs
            A token or a list of tokens.


        Returns
        -------
        mxnet.ndarray.NDArray:
            The embedding vector(s) of the token(s). According to numpy
            conventions, if `tokens` is a string, returns a 1-D NDArray of shape
            `self.vec_len`; if `tokens` is a list of strings, returns a 2-D
            NDArray of shape=(len(tokens), self.vec_len).
        """
        to_reduce = False
        if not isinstance(tokens, list):
            tokens = [tokens]
            to_reduce = True

        indices = [self.token_to_idx[token] if token in self.token_to_idx
                   else self.unknown_idx for token in tokens]

        vecs = nd.Embedding(nd.array(indices), self.idx_to_vec,
                            self.idx_to_vec.shape[0], self.idx_to_vec.shape[1])

        return vecs[0] if to_reduce else vecs

    def update_token_vectors(self, tokens, new_vectors):
        """Updates embedding vectors for tokens.


        Parameters
        ----------
        tokens : str or a list of strs.
            A token or a list of tokens whose embedding vector are to be
            updated.
        new_vectors : mxnet.ndarray.NDArray
            A 2-D NDArray to be assigned to the embedding vectors of `tokens`.
            Its length must be equal to the number of `tokens` and its width
            must be equal to the dimension of embeddings of the glossary.
        """

        assert self.idx_to_vec is not None, \
            'The property idx_to_vec has not been properly set.'

        if not isinstance(tokens, list):
            tokens = [tokens]

        assert isinstance(new_vectors, nd.NDArray) and \
               len(new_vectors.shape) == 2, 'new_vectors must be a 2-D NDArray.'
        assert new_vectors.shape[0] == len(tokens), \
            'The length of new_vectors must be equal to the number of tokens.'
        assert new_vectors.shape[1] == self.vec_len, \
            'The width of new_vectors must be equal to the dimension of ' \
            'embeddings of the glossary.'

        indices = []
        for token in tokens:
            if token in self.token_to_idx:
                indices.append(self.token_to_idx[token])
            else:
                raise ValueError('Token %s is unknown. To update the embedding '
                                 'vector for an unknown token, please specify '
                                 'it explicitly as the `unknown_token` %s in '
                                 '`tokens`. This is to avoid unintended '
                                 'updates.' %
                                 (token, self.idx_to_token[self.unknown_idx]))

        self._idx_to_vec[nd.array(indices)] = new_vectors

    @staticmethod
    def register(embed_cls):
        """Registers a new text embedding.

        Once an embedding is registered, we can create an instance of this
        embedding with `create_embedding` later.


        Examples
        --------
        >>> @mxnet.text.embedding.TextEmbed.register
        ... class MyTextEmbed(mxnet.text.glossary.TextEmbed):
        ...     def __init__(self, pretrain_file='my_pretrain_file'):
        ...         pass
        >>> embed = mxnet.text.embedding.TextEmbed.create('MyTextEmbed')
        >>> print(type(embed))
        <class '__main__.MyTextEmbed'>
        """

        assert(isinstance(embed_cls, type))
        embed_name = embed_cls.__name__.lower()
        if embed_name in TextEmbed.embed_registry:
            warnings.warn('New embedding %s.%s is overriding existing '
                          'embedding %s.%s', embed_cls.__module__,
                          embed_cls.__name__,
                          TextEmbed.embed_registry[embed_name].__module__,
                          TextEmbed.embed_registry[embed_name].__name__)
        TextEmbed.embed_registry[embed_name] = embed_cls
        return embed_cls

    @staticmethod
    def create(embed_name, **kwargs):
        """Creates an TextEmbed instance.

        Creates a text embedding instance by loading embeddings from an
        externally hosted pre-trained file, such as pre-trained embedding files
        of GloVe and FastText. To get all the available `embed_name` and
        `pretrain_file`, use TextEmbed.get_embed_names_and_pretrain_files().


        Parameters
        ----------
        embed_name : str
            The text embedding name (case-insensitive).


        Returns
        -------
        mxnet.text.glossary.TextEmbed:
            An embedding instance that loads embeddings from an externally
            hosted pre-trained file.
        """
        if embed_name.lower() in TextEmbed.embed_registry:
            return TextEmbed.embed_registry[embed_name.lower()](**kwargs)
        else:
            raise ValueError('Cannot find embedding %s. Valid embedding '
                             'names: %s' %
                             (embed_name,
                              ', '.join(TextEmbed.embed_registry.keys())))

    @staticmethod
    def check_pretrain_files(pretrain_file, embed_name):
        """Checks if a pre-trained file name is valid for an embedding.


        Parameters
        ----------
        pretrain_file : str
            The pre-trained file name.
        embed_name : str
            The text embedding name (case-insensitive).
        """
        embed_name = embed_name.lower()
        embed_cls = TextEmbed.embed_registry[embed_name]
        if pretrain_file not in embed_cls.pretrain_file_sha1:
            raise KeyError('Cannot find pretrain file %s for embedding %s. '
                           'Valid pretrain files for embedding %s: %s' %
                           (pretrain_file, embed_name, embed_name,
                            ', '.join(embed_cls.pretrain_file_sha1.keys())))

    @staticmethod
    def get_embed_names_and_pretrain_files():
        """Get valid TextEmbed names and pre-trained embedding file names.

        To load text embeddings from an externally hosted pre-trained file, such
        as pre-trained embedding files of GloVe and FastText, one should use
        TextEmbed.create(embed_name, pretrain_file). This method
        gets all the available `embed_name` and `pretrain_file`.


        Returns
        -------
        str:
            A string representation for all the available text embedding names
            (`embed_name`) and pre-trained embedding file names
            (`pretrain_file`). They can be plugged into
            `TextEmbed.create(embed_name, pretrain_file)`.
        """
        str_lst = []
        for embed_name, embed_cls in TextEmbed.embed_registry.items():
            str_lst.append('embed_name: %s\n' % embed_name)
            str_lst.append('pretrain_file: %s\n\n' %
                           ', '.join(embed_cls.pretrain_file_sha1.keys()))
        return ''.join(str_lst)


@TextEmbed.register
class GloVe(TextEmbed):
    """The GloVe text embedding.

    GloVe is an unsupervised learning algorithm for obtaining vector
    representations for words. Training is performed on aggregated global
    word-word co-occurrence statistics from a corpus, and the resulting
    representations showcase interesting linear substructures of the word vector
    space. (Source from https://nlp.stanford.edu/projects/glove/)

    Reference:
    GloVe: Global Vectors for Word Representation
    Jeffrey Pennington, Richard Socher, and Christopher D. Manning
    https://nlp.stanford.edu/pubs/glove.pdf

    Website:
    https://nlp.stanford.edu/projects/glove/

    To get the updated URLs to the externally hosted pre-trained text embedding
    files, visit https://nlp.stanford.edu/projects/glove/
    """

    # Map a pre-trained archive file and its SHA-1 hash.
    pretrain_archive_sha1 = \
        {'glove.42B.300d.zip': 'f8e722b39578f776927465b71b231bae2ae8776a',
         'glove.6B.zip': 'b64e54f1877d2f735bdd000c1d7d771e25c7dfdc',
         'glove.840B.300d.zip': '8084fbacc2dee3b1fd1ca4cc534cbfff3519ed0d',
         'glove.twitter.27B.zip': 'dce69c404025a8312c323197347695e81fd529fc'}

    # Map a pre-trained file and its SHA-1 hash.
    pretrain_file_sha1 = \
        {'glove.42B.300d.txt': '876767977d6bd4d947c0f84d44510677bc94612a',
         'glove.6B.50d.txt': '21bf566a9d27f84d253e0cd4d4be9dcc07976a6d',
         'glove.6B.100d.txt': '16b1dbfaf35476790bd9df40c83e2dfbd05312f1',
         'glove.6B.200d.txt': '17d0355ddaa253e298ede39877d1be70f99d9148',
         'glove.6B.300d.txt': '646443dd885090927f8215ecf7a677e9f703858d',
         'glove.840B.300d.txt': '294b9f37fa64cce31f9ebb409c266fc379527708',
         'glove.twitter.27B.25d.txt':
             '767d80889d8c8a22ae7cd25e09d0650a6ff0a502',
         'glove.twitter.27B.50d.txt':
             '9585f4be97e286339bf0112d0d3aa7c15a3e864d',
         'glove.twitter.27B.100d.txt':
             '1bbeab8323c72332bd46ada0fc3c99f2faaa8ca8',
         'glove.twitter.27B.200d.txt':
             '7921c77a53aa5977b1d9ce3a7c4430cbd9d1207a'}

    url_prefix = 'http://nlp.stanford.edu/data/'

    def __init__(self, pretrain_file='glove.840B.300d.txt',
                 embed_root='~/.mxnet/embeddings/', **kwargs):

        TextEmbed.check_pretrain_files(pretrain_file, GloVe.__name__)

        src_archive = {archive.split('.')[1]: archive for archive in
                       GloVe.pretrain_archive_sha1.keys()}
        archive = src_archive[pretrain_file.split('.')[1]]
        url = GloVe.url_prefix + archive

        super(GloVe, self).__init__(**kwargs)

        pretrain_file_path = TextEmbed._get_pretrain_file_path_from_url(
            url, embed_root, GloVe.__name__.lower(), pretrain_file)

        self._load_embedding(pretrain_file_path, ' ')


@TextEmbed.register
class FastText(TextEmbed):
    """The fastText text embedding.

    FastText is an open-source, free, lightweight library that allows users to
    learn text representations and text classifiers. It works on standard,
    generic hardware. Models can later be reduced in size to even fit on mobile
    devices. (Source from https://fasttext.cc/)

    References:
    Enriching Word Vectors with Subword Information
    Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov
    https://arxiv.org/abs/1607.04606

    Bag of Tricks for Efficient Text Classification
    Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov
    https://arxiv.org/abs/1607.01759

    FastText.zip: Compressing text classification models
    Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Herve Jegou,
    and Tomas Mikolov
    https://arxiv.org/abs/1612.03651

    Website:
    https://fasttext.cc/

    To get the updated URLs to the externally hosted pre-trained text embedding
    files, visit
    https://github.com/facebookresearch/fastText/blob/master/
    pretrained-vectors.md
    """

    # Map a pre-trained file and its SHA-1 hash.
    pretrain_file_sha1 = \
        {'wiki.en.vec': 'c1e418f144ceb332b4328d27addf508731fa87df',
         'wiki.simple.vec': '55267c50fbdf4e4ae0fbbda5c73830a379d68795',
         'wiki.zh.vec': '117ab34faa80e381641fbabf3a24bc8cfba44050'}
    url_prefix = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/'

    def __init__(self, pretrain_file='wiki.en.vec',
                 embed_root='~/.mxnet/embeddings/', **kwargs):

        TextEmbed.check_pretrain_files(pretrain_file, FastText.__name__)
        url = FastText.url_prefix + pretrain_file

        super(FastText, self).__init__(**kwargs)

        pretrain_file_path = TextEmbed._get_pretrain_file_path_from_url(
            url, embed_root, FastText.__name__.lower(), pretrain_file)

        self._load_embedding(pretrain_file_path, ' ')


class CustomEmbed(TextEmbed):
    """A user-defined text embedding.

    FastText is an open-source, free, lightweight library that allows users to
    learn text representations and text classifiers. It works on standard,
    generic hardware. Models can later be reduced in size to even fit on mobile
    devices. (Source from https://fasttext.cc/)
    """

    def __init__(self, pretrain_file_path, elem_delim=' ', **kwargs):
        super(CustomEmbed, self).__init__(**kwargs)
        self._load_embedding(pretrain_file_path, elem_delim)
