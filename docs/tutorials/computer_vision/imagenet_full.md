# Training Deep Net on 14 Million Images with a Single Computer

This tutorial describes how to train a neural network on a full ImageNet dataset [1] with 14,197,087 images in 21,841 classes. By using four GeForce GTX 980 desktop graphics cards on a single computer, we produced a state-of-art model in 8.5 days.

This problem presents several technical challenges:

- How to pack and store the massive data
- How to minimize the memory consumption of the network, so that we can use a network with more capacity than the one used for the ImageNet 1K dataset
- How to train the model fast

We've also released our pre-trained model for this full ImageNet dataset.

## Data Preprocessing
The raw full ImageNet dataset is more than 1 TB. Before training the network, we need to shuffle the images, and then load batches of images to feed the neural network. Before we describe how we did it, let’s do some calculations.

Assume that we have two good storage devices [2]:

```
    | Device                    | 4K Random Seek        | Sequential Seek |
    | ------------------------- | --------------------- | --------------- |
    | WD Black (HDD)            | 0.43 MB /s (110 IOPS) | 170 MB/s        |
    | Samsung 850 PRO (SSD)     | 40 MB/s (10,000 IOPS) | 550 MB/s        |
```

A very naive approach to loading from a list is random seeking. If we use this approach, we will spend 677 hours with an HDD or 6.7 hours with an SSD, respectively. This is for read only. Although the results for the SSD isn't bad, a 1 TB SSD is expensive.

Sequential seek is much faster than random seek. Loading by batch is a sequential action. But it can't perform sequential seek directly. We need to randomly shuffle the training data first, then pack it into a sequential binary package.

This is the solution used by most deep learning packages. However, unlike the ImageNet 1K dataset, we *can't* store the images in raw pixel format because that would require more than 1 TB of space. Instead, we need to pack the images into a compressed format.

To do this:

- Store the images in JPEG format, and then pack them into binary a record.
- Split the list, and pack several record files, instead of one file. This allows packing the images in distributed fashion, because we will be eventually be bounded by the I/O cost during packing. We need to make the package readable from several record files, which is not too hard.
This allows us to store the entire ImageNet dataset in approximately 250 GB.

After packing, along with the threaded buffer iterator, we can achieve an I/O speed of around 3,000 images/sec on a standard HDD.

## Training the Model


Now that we have data, we need to decide which network structure to use. We will use the Inception-BN [3]-style model, which compared to other models, such as VGG, has fewer parameters and fewer parameters simplified sync problems. Considering that our problem is much more challenging than the problem with 1K classes, we add suitable capacity to the original Inception-BN structure by increasing the size of the filter by a factor of 1.5 in the bottom layers of the original Inception-BN network.

However, this creates a challenge for GPU memory. Because the GTX 980 has only 4 GB of GPU RAM, we need to minimize memory consumption to fit larger batch sizes into training. To solve this problem, we use techniques such as node memory reuse and in-place optimization, which reduce memory consumption by half. For more details, see the [memory optimization note](http://mxnet.io/architecture/note_memory.html).

Finally, we can't train the model using a single GPU because this is a really large network and a lot of data. We use data parallelism on four GPUs for training, which involves smart synchronization of parameters between different GPUs, and overlap communication and computation. To simplify this task, we use a [runtime dependency engine](http://mxnet.io/architecture/note_engine.html), allowing us to run training at approximately 170 images/sec.

Here is a learning curve of the training process:
![alt text](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/imagenet_full/curve.png "Learning Curve")

## Evaluating Performance
Train Top-1 Accuracy over 21,841 classes: 37.19%

There is no official validation set over 21,841 classes, so we are using the ILVRC2012 validation set to check performance. Here's the result:

```
    | Accuracy | Over 1,000 classes | Over 21,841 classes |
    | -------- | ------------------ | ------------------- |
    | Top-1    | 68.3%              | 41.9%               |
    | Top-5    | 89.0%              | 69.6%               |
    | Top=20   | 96.0%              | 83.6%               |
```

We get reasonable results after nine iterations. Fewer iterations are needed to achieve stable performance, mainly because we have a larger dataset.

This result is by no means optimal. We didn't carefully pick the parameters, and the experiment cycle is longer than it was for the 1K dataset. There is definitely room for improvement. You are welcomed to try it out for yourself!


## The Code and Model
The code and guide are available on [GitHub](https://github.com/dmlc/mxnet/tree/master/example/image-classification). We also released a pretrained model on [GitHub](https://github.com/dmlc/mxnet-model-gallery/tree/master/imagenet-21k-inception.md).

## How to Use the Model
Training 21K classes are much more challenging than training 1K classes. It's impractical to use the raw prediction directly.

Look at this picture of Mount Rainier:

![alt text](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/imagenet_full/rainier.png "Mount Rainer")

We can detect that there is a mountain, valley, tree, and bridge. And the prediction probability is:

![alt text](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/imagenet_full/prob.png "Probability")

Notice that there are several peaks. Let's print out the label text for the ImageNet 1K classes and 21K classes:

```
    | Rank  | Over 1,000 classes          | Over 21,841 classes        |
    | ----- | --------------------------- | -------------------------- |
    | Top-1 | n09468604 valley            | n11620673 Fir              |
    | Top-2 | n09332890 lakeside          | n11624531 Spruce           |
    | Top-3 | n04366367 suspension bridge | n11621281 Amabilis fir     |
    | Top-4 | n09193705 alp               | n11628456 Douglas fir      |
    | Top-5 | n09428293 seashore          | n11627908 Mountain hemlock |
```

Prediction probability over 21K classes loses diversity. By carefully choosing a subset using WordNet hierarchy relation, you will get more interesting results.

## Notes
[1] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." *Computer Vision and Pattern Recognition*, 2009. CVPR 2009. IEEE Conference on. IEEE, 2009.

[2] HDD/SSD data is from a public website and might not be accurate.

[3] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *arXiv preprint arXiv:1502.03167* (2015).

## Next Steps
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)
