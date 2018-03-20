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
# pylint: disable=redefined-builtin

"""Utility functions."""

def flatten_samples(samples):
    """Flatten list of list of tokens into a single flattened list of tokens.

    Parameters
    ----------
    samples : list of list of object
        List of samples, each of which is a list of tokens.

    Returns
    -------
    Flattened list of tokens.
    """
    return [token for sample in samples for token in sample if token]

def collate(flat_sample, seq_len, overlap=0):
    """Collate a flat list of tokens into list of list of tokens, with each
    inner list's length equal to the specified `seq_len`.

    Parameters
    ----------
    flat_sample : list of object
        A flat list of tokens.
    seq_len : int
        The length of each of the samples.
    overlap : int, default 0
        The extra number of items in current sample that should overlap with the
        next sample.

    Returns
    -------
    List of samples, each of which has length equal to `seq_len`.
    """
    num_samples = (len(flat_sample)-seq_len) // (seq_len-overlap) + 1
    return [flat_sample[i*(seq_len-overlap):((i+1)*seq_len-i*overlap)] for i in range(num_samples)]

def collate_pad_length(num_items, seq_len, overlap=0):
    """Calculate the padding length needed for collated samples in order not to discard data.

    Parameters
    ----------
    num_items : int
        Number of items in dataset before collating.
    seq_len : int
        The length of each of the samples.
    overlap : int, default 0
        The extra number of items in current sample that should overlap with the
        next sample.

    Returns
    -------
    Length of paddings.
    """
    step = seq_len-overlap
    span = num_items-seq_len
    return (span // step + 1) * step - span
