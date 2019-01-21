# pre-trained-models

This shows examples of how to use the pretrained models. MXNet comes with a number of pretrained models
https://mxnet.incubator.apache.org/model_zoo/index.html


## Predict Image from pretrained models

From the example on https://mxnet.incubator.apache.org/tutorials/python/predict_image.html


The `predict-image.clj` file loads up the pre-trained resnet-152 model and uses it to predict the classifications from images on the internet

*To use run download-reset-152.sh to get the model params and json *

Run the example with the available leiningen alias:

```
$ lein predict-image
#
# or with your own image:
#
$ lein predict-image <url_or_path_to_image>
```


## Fine Tune from pretrained models

From the finetune example https://mxnet.incubator.apache.org/faq/finetune.html

The `fine-tune.clj` file loads up the samller resnet-50 model and adds a fine tune layer to reclassify the caltech iamge set

*To use run download-resnet-50.sh to get the model params and json and download-caltech.sh to get the pregenerated rec files*

You can run the fine tune example by doing `lein run` (cpu)

You can control the devices you run on by doing:

`lein run :cpu 2` - This will run on 2 cpu devices
`lein run :gpu 1` - This will run on 1 gpu device
`lein run :gpu 2` - This will run on 2 gpu devices



