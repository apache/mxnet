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

"""Utility functions applicable to all areas of code (tests, tutorials, gluon, etc)."""

__all__ = ['check_sha1', 'download', 'create_dir']

import errno
import logging
import time
import os
import hashlib

try:
    import requests
except ImportError:
    class requests_failed_to_import(object):
        pass
    requests = requests_failed_to_import


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.

    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def _download_retry_with_backoff(fname, url, max_attempts=4):
    """Downloads a url with backoff applied, automatically re-requesting after a failure.
    :param fname : str
        Name of the target file being downloaded.
    :param url : str
        URL to download.
    :param max_attempts : int, optional
        The number of requests to attempt before throwing a RequestException.
    :return:
        A streaming request object.
    """
    attempts = 1
    backoff_coef = 50.0
    while True:
        try:
            print('Downloading %s from %s...' % (fname, url))
            r = requests.get(url, stream=True)

            # If the remote server returned an error, raise a descriptive HTTPError.
            # For non-error http codes (e.g. 200, 206) do nothing.
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException:
            # Likely non-2** result, possibly timeout or redirection failure.
            attempts = attempts + 1
            if attempts > max_attempts:
                print('Downloading %s from %s, failed after #%d attempts' % (fname, url, attempts))
                raise

            # Apply backoff with default values borrowed from the popular Boto3 lib.
            time.sleep((backoff_coef * (2 ** attempts)) / 1000.0)


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download a given URL

    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    elif os.path.isdir(path):
        fname = os.path.join(path, url.split('/')[-1])
    else:
        fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        r = _download_retry_with_backoff(fname, url)
        with open(fname, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. '
                              'The repo may be outdated or download may be incomplete. '
                              'If the "repo_url" is overridden, consider switching to '
                              'the default repo.'.format(fname))

    return fname


def create_dir(dirname):
    if not os.path.exists(dirname):
        try:
            logging.info('create directory %s', dirname)
            os.makedirs(dirname)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise OSError('failed to create ' + dirname)
    return
