Classify Real-World Images with Pre-trained Model
=================================================
MXNet is a flexible and efficient deep learning framework. One of the cool thing that a deep learning
algorithm can do is to classify real world images.

In this example we will show how to use a pretrained Inception-BatchNorm Network to predict the class of
real world image. The network architecture is decribed in [1].

The pre-trained Inception-BatchNorm network is able to be downloaded from [this link](http://webdocs.cs.ualberta.ca/~bx3/data/Inception.zip)
This model gives the recent state-of-art prediction accuracy on image net dataset.

Preface
-------
This tutorial is written in Rmarkdown.
- You can directly view the hosted version of the tutorial from [MXNet R Document](http://mxnet.readthedocs.org/en/latest/R-package/classifyRealImageWithPretrainedModel.html)
- You can find the download the Rmarkdown source from [here](https://github.com/dmlc/mxnet/blob/master/R-package/vignettes/classifyRealImageWithPretrainedModel.Rmd)

Pacakge Loading
---------------
To get started, we load the mxnet package by require mxnet.

```r
require(mxnet)
```

```
## Loading required package: mxnet
## Loading required package: methods
```

In this example, we also need the imager package to load and preprocess the images in R.


```r
require(imager)
```

```
## Loading required package: imager
## Loading required package: plyr
## Loading required package: magrittr
## Loading required package: stringr
## Loading required package: png
## Loading required package: jpeg
##
## Attaching package: 'imager'
##
## The following object is masked from 'package:magrittr':
##
##     add
##
## The following object is masked from 'package:plyr':
##
##     liply
##
## The following objects are masked from 'package:stats':
##
##     convolve, spectrum
##
## The following object is masked from 'package:graphics':
##
##     frame
##
## The following object is masked from 'package:base':
##
##     save.image
```

Load the Pretrained Model
-------------------------
Make sure you unzip the pre-trained model in current folder. And we can use the model
loading function to load the model into R.


```r
model = mx.model.load("Inception/Inception_BN", iteration=39)
```

We also need to load in the mean image, which is used for preprocessing using ```mx.nd.load```.


```r
mean.img = as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])
```

Load and Preprocess the Image
-----------------------------
Now we are ready to classify a real image. In this example, we simply take the parrots image
from imager package. But you can always change it to other images.

Load and plot the image:


```r
im <- load.image(system.file("extdata/parrots.png", package="imager"))
plot(im)
```

![plot of chunk unnamed-chunk-5](../../web-data/mxnet/knitr/classifyRealImageWithPretrainedModel-unnamed-chunk-5-1.png)

Before feeding the image to the deep net, we need to do some preprocessing
to make the image fit the input requirement of deepnet. The preprocessing
include cropping, and substraction of the mean.
Because mxnet is deeply integerated with R, we can do all the processing in R function.

The preprocessing function:


```r
preproc.image <- function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  xx <- floor((shape[1] - short.edge) / 2)
  yy <- floor((shape[2] - short.edge) / 2) 
  croped <- crop.borders(im, xx, yy)
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(croped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized) * 255
  dim(arr) <- c(224, 224, 3)
  # substract the mean
  normed <- arr - mean.img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}
```

We use the defined preprocessing function to get the normalized image.


```r
normed <- preproc.image(im, mean.img)
```

Classify the Image
------------------
Now we are ready to classify the image! We can use the predict function
to get the probability over classes.


```r
prob <- predict(model, X=normed)
dim(prob)
```

```
## [1] 1000    1
```

As you can see ```prob``` is a 1 times 1000 array, which gives the probability
over the 1000 image classes of the input.

We can use the ```max.col``` on the transpose of prob. get the class index.

```r
max.idx <- max.col(t(prob))
max.idx
```

```
## [1] 89
```

The index do not make too much sense. So let us see what it really corresponds to.
We can read the names of the classes from the following file.


```r
synsets <- readLines("Inception/synset.txt")
```

And let us see what it really is


```r
print(paste0("Predicted Top-class: ", synsets[[max.idx]]))
```

```
## [1] "Predicted Top-class: n01818515 macaw"
```

Actually I do not know what does the word mean when I saw it.
So I searched on the web to check it out.. and hmm it does get the right answer :)

Reference
---------
[1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).
