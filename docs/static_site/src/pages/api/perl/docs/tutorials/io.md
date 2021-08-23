---
layout: page_api
title: Data Loading API
is_tutorial: true
tag: perl
permalink: /api/perl/docs/tutorials/io
---
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

# Data Loading API

## Overview

A data iterator reads data batch by batch.

```perl
pdl> $data = mx->nd->ones([100,10])
pdl> $nd_iter = mx->io->NDArrayIter($data, batch_size=>25)
pdl> for my $batch (@{ $nd_iter }) { print $batch->data->[0],"\n" }
<AI::MXNet::NDArray 25x10 @cpu(0)>
<AI::MXNet::NDArray 25x10 @cpu(0)>
<AI::MXNet::NDArray 25x10 @cpu(0)>
<AI::MXNet::NDArray 25x10 @cpu(0)>
```

If `$nd_iter->reset()` is called, then reads the data again from beginning.

In addition, an iterator provides information about the batch, including the
shapes and name.

```perl
pdl> $nd_iter = mx->io->NDArrayIter(data=>{data => mx->nd->ones([100,10])}, label=>{softmax_label => mx->nd->ones([100])}, batch_size=>25)
pdl> print($nd_iter->provide_data->[0],"\n")
DataDesc[data,25x10,float32,NCHW]
pdl> print($nd_iter->provide_label->[0],"\n")
DataDesc[softmax_label,25,float32,NCHW]
```

So this iterator can be used to train a symbol whose input data variable has
name `data` and input label variable has name `softmax_label`.

## Predefined Data iterators

```perl
mx->io->NDArrayIter
mx->io->CSVIter
mx->io->ImageRecordIter
mx->io->ImageRecordInt8Iter
mx->io->ImageRecordUInt8Iter
mx->io->MNISTIter
mx->recordio->MXRecordIO
mx->recordio->MXIndexedRecordIO
mx->image->ImageIter
```

## Helper classes and functions

Data structures and other iterators provided in the `AI::MXNet::IO` package.

```perl
AI::MXNet::DataDesc
AI::MXNet::DataBatch
AI::MXNet::DataIter
AI::MXNet::ResizeIter
AI::MXNet::MXDataIter
```

A list of image modification functions provided by `AI::MXNet::Image`.

```perl
mx->image->imdecode
mx->image->scale_down
mx->image->resize_short
mx->image->fixed_crop
mx->image->random_crop
mx->image->center_crop
mx->image->color_normalize
mx->image->random_size_crop
mx->image->ResizeAug
mx->image->RandomCropAug
mx->image->RandomSizedCropAug
mx->image->CenterCropAug
mx->image->RandomOrderAug
mx->image->ColorJitterAug
mx->image->LightingAug
mx->image->ColorNormalizeAug
mx->image->HorizontalFlipAug
mx->image->CastAug
mx->image->CreateAugmenter
```

Functions to read and write RecordIO files.

```perl
mx->recordio->pack
mx->recordio->unpack
mx->recordio->unpack_img
```

## Develop a new iterator

Writing a new data iterator in Perl is straightforward. Most MXNet
training/inference programs accept an object with ``provide_data``
and ``provide_label`` properties.
Please refer to AI-MXNet/examples for the examples of custom iterators.
