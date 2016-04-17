# Training Deep Net on 14 Million Images by Using A Single Machine

This note describes how to train a neural network on Full ImageNet Dataset [1] with 14,197,087 images in 21,841 classes. **We achieved a state-of-art model by using 4 GeForce GTX 980 cards on a single machine in 8.5 days.**

There are several technical challenges in this problem.
1. How to pack and store the massive data.
2. How to minimize the memory consumption of the network, so we can use net with more capacity than those used for ImageNet 1K
3. How to train the model fast.

We also released our pre-trained model for this full ImageNet dataset.

## Data Preprocessing
The raw full ImageNet dataset is more than 1TB. Before training the network, we need to shuffle these images then load batch of images to feed the neural network. Before we describe how we solve it, let’s do some calculation first:

Assume we have two good storage device [2]:

```
| Device                    | 4K Random Seek        | Sequential Seek |
| ------------------------- | --------------------- | --------------- |
| WD Black (HDD)            | 0.43 MB /s (110 IOPS) | 170 MB/s        |
| Samsung 850 PRO (SSD)     | 40 MB/s (10,000 IOPS) | 550 MB/s        |
```

A very naive approach is loading from a list by random seeking. If use this approach, we will spend 677 hours with HDD or 6.7 hours with SSD respectively. This is only about read. Although SSD looks not bad, but 1TB SSD is not affordable for everyone.

But we notice sequential seek is much faster than random seek. Also, loading batch by batch is a sequential action. Can we make a change? The answer is we can't do sequential seek directly. We need random shuffle the training data first, then pack them into a sequential binary package.

This is the normal solution used by most deep learning packages. However, unlike ImageNet 1K dataset, where we ***cannot*** store the images in raw pixels format.  Because otherwise we will need more than 1TB space. Instead, we need to pack the images in compressed format.

***The key ingredients are***
- Store the images in jpeg format, and pack them into binary record.
- Split the list, and pack several record files, instead of one file.
   - This allows us to pack the images in distributed fashion, because we will be eventually bounded by the IO cost during packing.
   - We need to make the package being able to read from several record files, which is not too hard.
This will allow us to store the entire imagenet dataset in around 250G space.

After packing, together with threaded buffer iterator, we can simply achieve an IO speed of around 3,000 images/sec on a normal HDD.

## Training the model


Now we have data. We need to consider which network structure to use. We use Inception-BN [3] style model, compared to other models such as VGG, it has fewer parameters, less parameters simplified sync problem. Considering our problem is much more challenging than 1k classes problem, we add suitable capacity into original Inception-BN structure, by increasing the size of filter by factor of 1.5 in bottom layers of original Inception-BN network.
This however, creates a challenge for GPU memory. As GTX980 only have 4G of GPU RAM. We really need to minimize the memory consumption to fit larger batch-size into the training. To solve this problem we use the techniques such as node memory reuse, and inplace optimization, which reduces the memory consumption by half, more details can be found in  [memory optimization note](http://mxnet.readthedocs.org/en/latest/developer-guide/note_memory.html)

Finally, we cannot train the model using a single GPU because this is a really large net, and a lot of data. We use data parallelism on four GPUs to train this model, which involves smart synchronization of parameters between different GPUs, and overlap the communication and computation. A [runtime denpdency engine](https://mxnet.readthedocs.org/en/latest/developer-guide/note_engine.html) is used to simplify this task, allowing us to run the training at around 170 images/sec.

Here is a learning curve of the training process:
![alt text](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/imagenet_full/curve.png "Learning Curve")

## Evaluate the Performance
Train Top-1 Accuracy over 21,841 classes: 37.19%

There is no official validation set over 21,841 classes, so we are using ILVRC2012 validation set to check the performance. Here is the result:

```
| Accuracy | Over 1,000 classes | Over 21,841 classes |
| -------- | ------------------ | ------------------- |
| Top-1    | 68.3%              | 41.9%               |
| Top-5    | 89.0%              | 69.6%               |
| Top=20   | 96.0%              | 83.6%               |
```

As we can see we get quite reasonable result after 9 iterations. Notably much less number of iterations is needed to achieve a stable performance, mainly due to we are facing a larger dataset.

We should note that this result is by no means optimal, as we did not carefully pick the parameters and the experiment cycle is longer than the 1k dataset. We think there is definite space for improvement, and you are welcomed to try it out by yourself!


## The Code and Model
The code and step guide is publically available at [https://github.com/dmlc/mxnet/tree/master/example/image-classification](https://github.com/dmlc/mxnet/tree/master/example/image-classification)

We also release a pretrained model under [https://github.com/dmlc/mxnet-model-gallery/tree/master/imagenet-21k-inception](https://github.com/dmlc/mxnet-model-gallery/tree/master/imagenet-21k-inception)

## How to Use The Model
We should point out it 21k classes is much more challenging than 1k. Directly use the raw prediction is not a reasonable way.

Look at this picture which I took in Mount Rainier this summer:

![alt text](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/imagenet_full/rainier.png "Mount Rainer")

We can figure out there is a mountain, valley, tree and bridge. And the prediction probability is :

![alt text](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/imagenet_full/prob.png "Probability")

We notice there are several peaks. Let's print out the label text in among 21k classes and ImageNet 1k classes:

```
| Rank  | Over 1,000 classes          | Over 21,841 classes        |
| ----- | --------------------------- | -------------------------- |
| Top-1 | n09468604 valley            | n11620673 Fir              |
| Top-2 | n09332890 lakeside          | n11624531 Spruce           |
| Top-3 | n04366367 suspension bridge | n11621281 Amabilis fir     |
| Top-4 | n09193705 alp               | n11628456 Douglas fir      |
| Top-5 | n09428293 seashore          | n11627908 Mountain hemlock |
```

There is no doubt that directly use probability over 21k classes loss diversity of prediction. If you carefully choose a subset by using WordNet hierarchy relation, I am sure you will find more interesting results.

## Note
[1] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." *Computer Vision and Pattern Recognition*, 2009. CVPR 2009. IEEE Conference on. IEEE, 2009.

[2] HDD/SSD data is from public website may not be accurate.

[3] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *arXiv preprint arXiv:1502.03167* (2015).
