# Neural art

This is an implementation of the paper
[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon
A. Gatys, Alexander S. Ecker, and Matthias Bethge.

## How to use

First use `download.sh` to download pre-trained model and sample inputs

Then run `python run.py`, use `-h` to see more options

## Sample results

<img src=https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/output/4343_starry_night.jpg width=600px>

It takes 30 secs for a Titan X to generate the above 600x400 image.

## Note

* The current implementation is based the
  [torch implementation](https://github.com/jcjohnson/neural-style). But we may
  change it dramatically in the near future.
