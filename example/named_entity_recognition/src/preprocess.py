# !/usr/bin/env python

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

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#read in csv of NER training data
df = pd.read_csv("../data/ner_dataset.csv", encoding="ISO-8859-1")

#rename columns
df = df.rename(columns = {"Sentence #" : "utterance_id",
                            "Word" : "token", 
                            "POS" : "POS_tag", 
                            "Tag" : "BILOU_tag"})

#clean utterance_id column
df.loc[:, "utterance_id"] = df["utterance_id"].str.replace('Sentence: ', '')

#fill np.nan utterance ID's with the last valid entry
df = df.fillna(method='ffill')
df.loc[:, "utterance_id"] = df["utterance_id"].apply(int)

#melt BILOU tags and tokens into an array per utterance
df1 = df.groupby("utterance_id")["BILOU_tag"].apply(lambda x: np.array(x)).to_frame().reset_index()
df2 = df.groupby("utterance_id")["token"].apply(lambda x: np.array(x)).to_frame().reset_index()
df3 = df.groupby("utterance_id")["POS_tag"].apply(lambda x: np.array(x)).to_frame().reset_index()

#join the results on utterance id
df = df1.merge(df2.merge(df3, how = "left", on = "utterance_id"), how = "left", on = "utterance_id")

#save the dataframe to a csv file
df.to_pickle("../data/ner_data.pkl")