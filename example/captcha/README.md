This is the R version of [captcha recognition](http://blog.xlvector.net/2016-05/mxnet-ocr-cnn/) example by xlvector and it can be used as an example of multi-label training. For a captcha below, we consider it as an image with 4 labels and train a CNN over the data set.

![](captcha_example.png)

You can download the images and `.rec` files from [here](https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/data/captcha_example.zip). Since each image has 4 labels, please remember to use `label_width=4` when generating the `.rec` files.
