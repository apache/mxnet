# Classify Images with a PreTrained Model



MXNet is a flexible and efficient deep learning framework. One of the interesting things that a deep learning algorithm can do is classify real world images. 

In this tutorial, we will show you how to leverage **mxnet** to load an out-of-the-box model that is capable of classifying objects in images, *without* requiring you to provide any training data of your own.
Specifically, we demonstrate how to use a pre-trained Inception-BatchNorm network for classifying an image into 1 out of 1000 common classes. At the time it was proposed, this model achieved state-of-art object recognition accuracy on the large-scale ImageNet dataset. 

For information about the model, see the paper:

[Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv:1502.03167 (2015)](http://arxiv.org/abs/1502.03167).


Let's now load some required R packages and download the pre-trained Inception-BatchNorm network:

```{.python .input  n=3}
require(mxnet)
require(imager) # for loading/processing images in R

if (!file.exists('Inception.zip')) {
    download.file(url='http://data.mxnet.io/mxnet/data/Inception.zip',
                  destfile='Inception.zip', method='wget')
}
if (!dir.exists('Inception')) {
    system("unzip Inception.zip")
}
```

The above code relies on the ``wget`` and ``unzip`` commands being installed on your machine. 
If it fails, you can instead manually the pre-trained Inception-BatchNorm network from [this link](http://data.mxnet.io/mxnet/data/Inception.zip). 
In this case, you *must* first download this ZIP file and unzip it inside the current working directory (enter ``getwd()`` in R console to determine which directory this is).
At this point, there should now be a sub-folder titled ``Inception/`` inside the current working directory.

## Load the Pre-trained Model

Use the provided model loading function to load the pre-trained neural network model into R:

```{.python .input  n=4}
model = mx.model.load("Inception/Inception_BN", iteration=39)
```

Also load in the mean image, which is needed for preprocessing:

```{.python .input  n=5}
mean.img = as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])
```

## Load and Preprocess the Image

Now, we are ready to classify a real image. In this example, we simply take the parrots image from the **imager** package. You can use another image, if you prefer. Load and plot the image:

```{.python .input  n=6}
im <- load.image(system.file("extdata/parrots.png", package="imager"))
plot(im)
```

Before feeding the image to the deep network, we need to perform some preprocessing to make the image meet the deep network input requirements. 
Preprocessing includes cropping of the image and subtracting the mean (so the distribution of each pixel value is centered around 0). 
Since **mxnet** is deeply integrated with R, we can do all the processing in the following R function:

```{.python .input  n=11}
preproc.image <- function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  xx <- floor((shape[1] - short.edge) / 2)
  yy <- floor((shape[2] - short.edge) / 2)
  cropped <- crop.borders(im, xx, yy)
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(cropped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized) * 255
  dim(arr) <- c(224, 224, 3)
  # subtract the mean
  normed <- arr - mean.img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}
```

Now use our preprocessing function to get a normalized version of the image:

```{.python .input  n=12}
normed <- preproc.image(im, mean.img)
```

## Classify the Image

Now we are ready to classify the image! 
Recall that the network we have loaded was previously trained to distinguish between 1000 different image classes.
Below, we use the ``predict`` function to get the class-probabilities predicted by our pre-trained network for this image. We can also use ``max.col`` to identify which class our network predicts to be the most likely for this image.

```{.python .input  n=14}
prob <- predict(model, X=normed)
max.idx <- max.col(t(prob))
max.idx
```

The index of the most-likely class doesn’t make much sense, so let’s see what it really means. 
We read the names of the classes from the following file:

```{.python .input  n=15}
synsets <- readLines("Inception/synset.txt")
```

Let’s see what the image really is:

```{.python .input  n=17}
print(paste("Predicted Top-class:", synsets[[max.idx]]))
```

It’s a macaw!

To see a list of other state-of-the-art pre-trained models that can also be loaded into **mxnet** to use in your own applications, check out the [Model Zoo](https://mxnet.apache.org/model_zoo/index.html).  MXNet provides pre-trained networks for text data in addition to images.
