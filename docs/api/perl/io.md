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


```perl
pdl> $data  = mx->sym->Variable('data')
pdl> $label = mx->sym->Variable('softmax_label')
pdl> $fullc = mx->sym->FullyConnected(data=>$data, num_hidden=>1)
pdl> $loss  = mx->sym->SoftmaxOutput(data=>$data, label=>$label)
pdl> $mod   = mx->mod->Module($loss)
pdl> print($mod->data_names->[0])
data
pdl> print($mod->label_names->[0])
softmax_label
pdl> $mod->bind(data_shapes=>$nd_iter->provide_data, label_shapes=>$nd_iter->provide_label)
```

Then we can call `$mod->fit($nd_iter, num_epoch=>2)` to train `loss` by 2 epochs.

## Predefined Data iterators

```perl
mx->io->NDArrayIter
mx->io->CSVIter
mx->io->ImageRecordIter
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
