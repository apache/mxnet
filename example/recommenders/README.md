<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Recommender Systems


This directory has a set of examples of how to build various kinds of recommender systems
using MXNet. The sparsity of user / item data is handled through the embedding layers that accept
indices as input rather than one-hot encoded vectors.


## Examples

The examples are driven by notebook files.

* [Matrix Factorization: linear and non-linear models](demo1-MF.ipynb)
* [Deep Structured Semantic Model (DSSM) for content-based recommendations](demo2-dssm.ipynb)


### Negative Sampling

* A previous version of this example had an example of negative sampling. For example of negative sampling, please refer to:
    [Gluon NLP Sampled Block](https://github.com/dmlc/gluon-nlp/blob/master/src/gluonnlp/model/sampled_block.py)
    

## Acknowledgements

Thanks to [xlvector](https://github.com/xlvector/) for the first Matrix Factorization example
that provided the basis for these examples.

[MovieLens](http://grouplens.org/datasets/movielens/) data from [GroupLens](http://grouplens.org/).
Note: MovieLens 100K and 10M dataset are copy right to GroupLens Research Group at the University of Minnesota,
and licensed under their usage license. For full text of the usage license, see [ml-100k license](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt)
 and [ml-10m license](http://files.grouplens.org/datasets/movielens/ml-10m-README.html). 