# Neural Art


This is an implementation of the paper
[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon
A. Gatys, Alexander S. Ecker, and Matthias Bethge.

Get the source code for this tutorial from [GitHub](https://github.com/dmlc/mxnet/tree/master/example/neural-style).

The current implementation is based on the
  [torch implementation](https://github.com/jcjohnson/neural-style), but we might
  change it dramatically in the near future. We will release a multi-GPU version soon.

## Run the Model

1. Download the trained model and sample inputs using `download.sh`. 

1. Run `python nstyle.py`. To see more options, use `-h`.

## Sample Results

<img src=https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/output/4343_starry_night.jpg width=600px>

It takes 30 seconds for a Titan X to generate this 600x400 image.


## Next Steps
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)