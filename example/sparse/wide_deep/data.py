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

# pylint: skip-file
from csv import DictReader
import os
import mxnet as mx
import numpy as np


def get_uci_adult(data_dir, data_name, url):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    os.chdir(data_dir)
    if (not os.path.exists(data_name)):
        print("Dataset " + data_name + " not present. Downloading now ...")
        os.system("wget %r" % url + data_name)
        if "test" in data_name:
            os.system("sed -i '1d' %r" % data_name)
        print("Dataset " + data_name + " is now present.")
    csr, dns, label = preprocess_uci_adult(data_name)
    os.chdir("..")
    return csr, dns, label


def preprocess_uci_adult(data_name):
    """Some tricks of feature engineering are adapted
    from tensorflow's wide and deep tutorial.
    """
    csv_columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "gender",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income_bracket"
    ]

    vocabulary_dict = {
        "gender": [
            "Female", "Male"
        ],
        "education": [
            "Bachelors", "HS-grad", "11th", "Masters", "9th",
            "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
            "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
            "Preschool", "12th"
        ],
        "marital_status": [
            "Married-civ-spouse", "Divorced", "Married-spouse-absent",
            "Never-married", "Separated", "Married-AF-spouse", "Widowed"
        ],
        "relationship": [
            "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
            "Other-relative"
        ],
        "workclass": [
            "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
            "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
        ]
    }
    # wide columns
    crossed_columns = [
        ["education", "occupation"],
        ["native_country", "occupation"],
        ["age_buckets", "education", "occupation"],
    ]
    age_boundaries = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    # deep columns
    indicator_columns = ['workclass', 'education', 'gender', 'relationship']
    
    embedding_columns = ['native_country', 'occupation']

    continuous_columns = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    # income_bracket column is the label
    labels = ["<", ">"]

    hash_bucket_size = 1000
    
    csr_ncols = len(crossed_columns) * hash_bucket_size
    dns_ncols = len(continuous_columns) + len(embedding_columns)
    for col in indicator_columns:
        dns_ncols += len(vocabulary_dict[col])

    label_list = []
    csr_list = []
    dns_list = []

    with open(data_name) as f:
        for row in DictReader(f, fieldnames=csv_columns):
            label_list.append(labels.index(row['income_bracket'].strip()[0]))

            for i, cols in enumerate(crossed_columns):
                if cols[0] == "age_buckets":
                    age_bucket = np.digitize(float(row["age"]), age_boundaries)
                    s = '_'.join([row[col].strip() for col in cols[1:]])
                    s += '_' + str(age_bucket)
                    csr_list.append((i * hash_bucket_size + hash(s) % hash_bucket_size, 1.0))
                else:
                    s = '_'.join([row[col].strip() for col in cols])
                    csr_list.append((i * hash_bucket_size + hash(s) % hash_bucket_size, 1.0))
            
            dns_row = [0] * dns_ncols
            dns_dim = 0
            for col in embedding_columns:
                dns_row[dns_dim] = hash(row[col].strip()) % hash_bucket_size
                dns_dim += 1

            for col in indicator_columns:
                dns_row[dns_dim + vocabulary_dict[col].index(row[col].strip())] = 1.0
                dns_dim += len(vocabulary_dict[col])

            for col in continuous_columns:
                dns_row[dns_dim] = float(row[col].strip())
                dns_dim += 1

            dns_list.append(dns_row)

    data_list = [item[1] for item in csr_list]
    indices_list = [item[0] for item in csr_list]
    indptr_list = range(0, len(indices_list) + 1, len(crossed_columns))
    # convert to ndarrays
    csr = mx.nd.sparse.csr_matrix((data_list, indices_list, indptr_list),
                                  shape=(len(label_list), hash_bucket_size * len(crossed_columns)))
    dns = np.array(dns_list)
    label = np.array(label_list)
    return csr, dns, label
