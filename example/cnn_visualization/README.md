# Visualzing CNN decisions

This folder contains an MXNet Gluon implementation of [Grad-CAM](https://arxiv.org/abs/1610.02391) that helps visualize CNN decisions.

A tutorial on how to use this from Jupyter notebook is available [here](https://mxnet.incubator.apache.org/tutorials/vision/cnn_visualization.html).

You can also do the visualization from terminal:
```
$ python gradcam_demo.py hummingbird.jpg
Predicted category  : hummingbird (94)
Original Image      : hummingbird_orig.jpg
Grad-CAM            : hummingbird_gradcam.jpg
Guided Grad-CAM     : hummingbird_guided_gradcam.jpg
Saliency Map        : hummingbird_saliency.jpg
```

![Output of gradcam_demo.py](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/cnn_visualization/hummingbird_filenames.png)
