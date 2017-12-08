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

"""Read text files and load embeddings."""
from __future__ import absolute_import
from __future__ import print_function

import logging
import os
import tarfile
import zipfile

from ..gluon.utils import check_sha1
from ..gluon.utils import download
from .. import ndarray as nd

from tqdm import tqdm


class Glossary(object):
    """Indexing and embedding for text and special tokens in a glossary.

    For each indexed text or special token (e.g., an unknown token) in a
    glossary, an embedding vector will be associated with the token. Such
    embedding vectors can be loaded from externally pre-trained embeddings,
    such as via mxnet.text.Embedding instances.


    Parameters
    ----------
    counter : collections.Counter
        Counts text token frequencies in the text data.
    top_k_freq : None or int, default None
        The number of top frequent tokens in the keys of `counter` that will be
        indexed. If None, all the tokens in the keys of `counter` will be
        indexed.
    min_freq : int, default 1
        The minimum frequency required for a token in the keys of `counter` to
        be indexed.
    specials : list of strs, default ['<unk>']
        A list of special tokens to be indexed. It must be an non-empty list
        whose first element is the string representation for unknown tokens,
        such as '<unk>'. It cannot contain any token from the keys of `counter`.
    embeds : an mxnet.text.Embedding instance, a list of mxnet.text.Embedding
             instances, or None, default None
        Pre-trained embeddings to load from. If None, there is nothing to load.


    Properties
    ----------
    counter : collections.Counter
        Counts text and special token frequencies in the text data, where
        special token frequency is clear to zero.
    token_to_idx : dict
        A dict mapping each token to its index integer.
    idx_to_token : list
        A list of indexed tokens where the list indices and the token indices
        are aligned.
    idx_to_vec : mxnet.ndarray.NDArray
        For all the indexed tokens in this glossary, this NDArray maps each
        token's index to an embedding vector.
    vec_len : int
        The length of the embedding vector for any token.
    specials:
        A list of special tokens to be indexed. It must be an non-empty list
        whose first element is the string representation for unknown tokens,
        such as '<unk>'. It excludes all the tokens from the keys of `counter`.
    """
    def __init__(self, counter, top_k_freq=None, min_freq=1,
                 specials=['<unk>'], embeds=None):
        # Sanity checks.
        assert min_freq > 0, 'min_freq must be set to a positive value.'
        assert len(specials) > 0, \
            'specials must be an non-empty list whose first element is the ' \
            'string representation for unknown tokens, such as "<unk>".'

        # Initialize attributes.
        self._init_attrs(counter, specials)

        # Set self._idx_to_token and self._token_to_idx according to specified
        # frequency thresholds.
        self._set_idx_and_token(counter, specials, top_k_freq, min_freq)

        if embeds is not None:
            self.set_idx_to_vec(embeds)

    def _init_attrs(self, counter, specials):
        self._counter = counter.copy()
        self._token_to_idx = {token: idx for idx, token in enumerate(specials)}
        self._idx_to_token = specials.copy()
        self._idx_to_vec = None
        self._vec_len = 0
        self._specials = specials.copy()

    def _set_idx_and_token(self, counter, specials, top_k_freq, min_freq):
        # Update _counter to include special specials, such as '<unk>'.
        self._counter.update({token: 0 for token in specials})
        assert len(self._counter) == len(counter) + len(specials), 'specials ' \
            'cannot contain any token from the keys of `counter`.'

        token_freqs = sorted(self._counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)

        token_cap = len(self._counter) if top_k_freq is None \
            else len(specials) + top_k_freq

        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) == token_cap:
                break
            self._idx_to_token.append(token)
            self._token_to_idx[token] = len(self._idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    @property
    def counter(self):
        return self._counter

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def idx_to_vec(self):
        return self._idx_to_vec

    @property
    def vec_len(self):
        return self._vec_len

    @property
    def specials(self):
        return self._specials

    @staticmethod
    def unk_idx():
        return 0

    def set_idx_to_vec(self, embeds):
        # Sanity check.
        if isinstance(embeds, list):
            for loaded_embed in embeds:
                assert isinstance(loaded_embed, Embedding)
        else:
            assert isinstance(embeds, Embedding)
            embeds = [embeds]

        self._vec_len = sum(embed.vec_len for embed in embeds)
        self._idx_to_vec = nd.zeros(shape=(len(self), self.vec_len))

        col_start = 0
        # Concatenate all the embedding vectors in embeds.
        for embed in embeds:
            col_end = col_start + embed.vec_len
            self._idx_to_vec[:, col_start:col_end] = embed[self.idx_to_token]
            col_start = col_end

    # tokens is a str or a list of strs.
    def update_idx_to_vec(self, tokens, new_vectors):
        assert self.idx_to_vec is not None, \
            'mxnet.text.Glossary._idx_to_vec has not been initialized.  Use ' \
            'mxnet.text.Glossary.__init__() or ' \
            'mxnet.text.Glossary.set_idx_to_embed() to initialize it.'

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
                raise ValueError('Token %s is unknown to the glossary. To '
                                 'update the embedding vector for an unknown '
                                 'token, please specify it explicitly as the '
                                 'unknown special token %s in tokens. This is '
                                 'to avoid unintended updates.' %
                                 (token, self.idx_to_token[Glossary.unk_idx()]))
        self._idx_to_vec[nd.array(indices)] = new_vectors


class Embedding(object):

    # Key-value pairs for embedding name in lower case and embedding class.
    embed_registry = {}

    def __init__(self, pretrain_file, url=None, embed_name='my_embed',
                 embed_root='~/.mxnet/embeddings/', special_init_vec=nd.zeros,
                 token_delim=' '):

        pretrain_file = os.path.expanduser(pretrain_file)
        embed_root = os.path.expanduser(embed_root)

        # Sanity check.
        if not os.path.isfile(pretrain_file) and url is None:
            raise ValueError('The source pretrained embedding file to load '
                             'must be provided by the user via a valid path as '
                             'specified in pretrain_file or downloaded via '
                             'url.')
        if url is not None:
            assert embed_name in Embedding.embed_registry, \
                'To load a pretrained embedding file from url, use ' \
                'mxnet.text.Embedding.create_embedding(embed_name, **kwargs) ' \
                'or manually download it and specify its path via pretrain_' \
                'file.'
            assert not os.path.isfile(pretrain_file), \
                'When pretrain_file is a path to a user-provided pretrained ' \
                'embedding file, url must be set to None; when url to the ' \
                'pretrained embedding file(s) is specified, such as when ' \
                'embedding is created by ' \
                'mxnet.text.Embedding.create_embedding(embed_name, ' \
                '**kwargs), pretrain_file must be the name rather than the ' \
                'path of the pretrained embedding file. This is to avoid ' \
                'confusion over the source pretrained embedding file to ' \
                'load. Call ' \
                'mxnet.text.Embedding.show_embed_names_and_pretrain_files() ' \
                'to see the available embed_name and pretrain_file.'

        self._special_init_vec = special_init_vec

        pretrain_file_path = Embedding._get_pretrain_file(pretrain_file, url,
                                                          embed_name,
                                                          embed_root)

        self._load_embedding(pretrain_file_path, token_delim)

        assert len(self._idx_to_vec) - 1 == len(self._idx_to_token) \
            == len(self._token_to_idx), \
            'The extra (last) row of self._idx_to_vec is the initialized ' \
            'embedding vector for a special, such as an unknown token.'

    @staticmethod
    def _get_pretrain_file(pretrain_file, url, embed_name, embed_root):
        # User specifies pretrained embedding file at the path of pretrain_file.
        if os.path.isfile(pretrain_file):
            pretrain_file_path = pretrain_file

        # The pretrained embedding file will be downloaded from url if it has
        # not been downloaded yet or the existing file fails to match the
        # expected SHA-1 hash.
        else:
            embed_dir = os.path.join(embed_root, embed_name)
            pretrain_file_path = os.path.join(embed_dir, pretrain_file)
            download_file = os.path.basename(url)
            download_file_path = os.path.join(embed_dir, download_file)

            embed_cls = Embedding.embed_registry[embed_name]
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
                'mxnet.text.Embedding.create_embedding(%s, **kwargs) to load ' \
                'the pretrained embedding %s. If you still wish to load the ' \
                'changed embedding file, please specify its path %s via ' \
                'pretrain_file of mxnet.text.Embedding(). It will be loaded ' \
                'only if its data format remains unchanged.' % \
                (download_file_path, embed_name, embed_name, download_file_path)

            ext = os.path.splitext(download_file)[1]
            if ext == '.zip':
                with zipfile.ZipFile(download_file_path, 'r') as zf:
                    zf.extractall(embed_dir)
            elif ext == '.gz':
                with tarfile.open(download_file_path, 'r:gz') as tar:
                    tar.extractall(path=embed_dir)
        return pretrain_file_path

    def _load_embedding(self, pretrain_file_path, token_delim):
        with open(pretrain_file_path, 'r', encoding='utf8') as f:
            lines = f.readlines()

        logging.info('Loading pretrained embedding vectors from %s'
                     % pretrain_file_path)

        vec_len = None
        all_elems = []
        idx_to_token = []
        for line in tqdm(lines, total=len(lines)):
            elems = line.rstrip().split(token_delim)

            assert len(elems) > 1, 'The data format of the pretrained ' \
                                   'embedding file %s is unexpected.' \
                                   % pretrain_file_path

            token, elems = elems[0].decode('utf8'), [float(i)
                                                     for i in elems[1:]]

            if len(elems) == 1:
                logging.warning('WARNING: Token %s with 1-dimensional vector '
                                '%s is likely a header and is skipped.'
                                % (token, elems))
                continue
            else:
                if vec_len is None:
                    vec_len = len(elems)
                else:
                    assert len(elems) == vec_len, \
                        'The dimension of token %s is %d but the dimension ' \
                        'of previous tokens is %d. Dimensions of all the ' \
                        'tokens must be the same.' % (token, len(elems),
                                                      vec_len)

            all_elems.extend(elems)
            idx_to_token.append(token)

        self._vec_len = vec_len
        self._idx_to_token = idx_to_token
        self._token_to_idx = {token: idx for idx, token in
                              enumerate(self.idx_to_token)}

        all_elems.extend([0] * self.vec_len)
        self._idx_to_vec = nd.array(all_elems).reshape((-1, self.vec_len))
        self._idx_to_vec[-1] = self.special_init_vec(shape=self.vec_len)

    def __len__(self):
        return len(self.idx_to_token)

    @property
    def vec_len(self):
        return self._vec_len

    @property
    def special_init_vec(self):
        return self._special_init_vec

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def idx_to_vec(self):
        return self._idx_to_vec

    def _idx_to_vec_special_idx(self):
        return len(self._idx_to_vec) - 1

    # tokens is a str or a list of strs
    def __getitem__(self, tokens):
        to_reduce = False
        if not isinstance(tokens, list):
            tokens = [tokens]
            to_reduce = True

        indices = [self.token_to_idx[token] if token in self.token_to_idx
                   else self._idx_to_vec_special_idx() for token in tokens]

        vecs = nd.Embedding(nd.array(indices), self.idx_to_vec,
                            self.idx_to_vec.shape[0], self.idx_to_vec.shape[1])
        # By numpy conventions:
        # Embedding['token'] returns 1-D NDArray of shape=vec_len;
        # Embedding[['token']] returns 2-D NDArray of shape=(1, vec_len).
        return vecs[0] if to_reduce else vecs

    @staticmethod
    def register(embed_cls):
        assert(isinstance(embed_cls, type))
        embed_name = embed_cls.__name__.lower()
        if embed_name in Embedding.embed_registry:
            logging.warning('WARNING: New embedding %s.%s is overriding '
                            'existing embedding %s.%s',
                            embed_cls.__module__, embed_cls.__name__,
                            Embedding.embed_registry[embed_name].__module__,
                            Embedding.embed_registry[embed_name].__name__)
        Embedding.embed_registry[embed_name] = embed_cls
        return embed_cls

    @staticmethod
    def create_embedding(embed_name, **kwargs):
        if embed_name.lower() in Embedding.embed_registry:
            return Embedding.embed_registry[embed_name.lower()](**kwargs)
        else:
            raise ValueError('Cannot find embedding %s. Valid embedding '
                             'names: %s' % (embed_name, ', '.join(
                                Embedding.embed_registry.keys())))

    @staticmethod
    def check_pretrain_files(pretrain_file, embed_name):
        embed_cls = Embedding.embed_registry[embed_name]
        if pretrain_file not in embed_cls.pretrain_file_sha1:
            raise KeyError('Cannot find pretrain file %s for embedding %s. '
                           'Valid pretrain files for embedding %s: %s' %
                           (pretrain_file, embed_name, embed_name,
                            ', '.join(embed_cls.pretrain_file_sha1.keys())))

    @staticmethod
    def show_embed_names_and_pretrain_files():
        for embed_name, embed_cls in Embedding.embed_registry.items():
            print('embed_name: %s' % embed_name)
            print('pretrain_file: %s\n' %
                  ', '.join(embed_cls.pretrain_file_sha1.keys()))


@Embedding.register
class GloVe(Embedding):
    pretrain_archive_sha1 = \
        {'glove.42B.300d.zip': 'f8e722b39578f776927465b71b231bae2ae8776a',
         'glove.6B.zip': 'b64e54f1877d2f735bdd000c1d7d771e25c7dfdc',
         'glove.840B.300d.zip': '8084fbacc2dee3b1fd1ca4cc534cbfff3519ed0d',
         'glove.twitter.27B.zip': 'dce69c404025a8312c323197347695e81fd529fc'
        }
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
             '7921c77a53aa5977b1d9ce3a7c4430cbd9d1207a'
        }
    url_prefix = 'http://nlp.stanford.edu/data/'

    def __init__(self, pretrain_file='glove.840B.300d.txt', **kwargs):
        cls = self.__class__

        Embedding.check_pretrain_files(pretrain_file, cls.__name__.lower())

        src_archive = {archive.split('.')[1]: archive for archive in
                       cls.pretrain_archive_sha1.keys()}
        archive = src_archive[pretrain_file.split('.')[1]]
        url = cls.url_prefix + archive

        super(cls, self).__init__(pretrain_file=pretrain_file, url=url,
                                  embed_name=cls.__name__.lower(), **kwargs)


@Embedding.register
class FastText(Embedding):
    pretrain_file_sha1 = \
        {'wiki.en.vec': 'c1e418f144ceb332b4328d27addf508731fa87df',
         'wiki.simple.vec': '55267c50fbdf4e4ae0fbbda5c73830a379d68795',
         'wiki.zh.vec': '117ab34faa80e381641fbabf3a24bc8cfba44050'}
    url_prefix = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/'

    def __init__(self, pretrain_file='wiki.en.vec', **kwargs):
        cls = self.__class__

        Embedding.check_pretrain_files(pretrain_file, cls.__name__.lower())
        url = cls.url_prefix + pretrain_file

        super(cls, self).__init__(pretrain_file=pretrain_file, url=url,
                                  embed_name=cls.__name__.lower(), **kwargs)
