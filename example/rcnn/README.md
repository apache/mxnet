# Faster R-CNN in MXNet with distributed implementation and data parallelization

Region Proposal Network solves object detection as a regression problem 
from the objectness perspective. Bounding boxes are predicted by applying 
learned bounding box deltas to base boxes, namely anchor boxes across 
different positions in feature maps. Training process directly learns a 
mapping from raw image intensities to bounding box transformation targets.

Fast R-CNN treats general object detection as a classification problem and
bounding box prediction as a regression problem. Classifying cropped region
feature maps and predicting bounding box displacements together yields
detection results. Cropping feature maps instead of image input accelerates
computation utilizing shared convolution maps. Bounding box displacements
are simultaneously learned in the training process.

Faster R-CNN utilize an alternate optimization training process between RPN 
and Fast R-CNN. Fast R-CNN weights are used to initiate RPN for training.

## Getting Started

* Install a forked MXNet at [MXNet-detection](https://github.com/precedenceguo/mxnet/tree/detection).
Follow the instructions at http://mxnet.readthedocs.io/en/latest/how_to/build.html. Install the python interface.
Note that the link refers to `detection` branch of the fork. Use `git clone -b detection https://github.com/precedenceguo/mxnet.git`
to clone or `git checkout detection` if you checked out the master.
* Download data and place them to `data` folder according to `Data Folder Structure`.
  You might want to create a symbolic link to VOCdevkit folder
```
Pascal VOCdevkit
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```
* Data Folder Structure (suppose root is `data`)
```
demo
rpn_data (created by rpn)
selective_search_data (can be omitted)
cache (created by imdb)
-- name + source + roidb.pkl (create by imdb)
-- name (created by detection and evaluation)
VOCdevkit
-- VOC + year (JPEG images and annotations)
-- results (created by evaluation)
---- VOC + year
------ main
-------- comp4_det_val_aeroplane.txt
```
* Download VGG16 pretrained model, use `mxnet/tools/caffe_converter` to convert it,
  rename to `vgg16-symbol.json` and `vgg16-0001.params` and place it in `model` folder

## Training
* Start training by run `python -m tools.train_alternate`. Variable args can be found by run
`python -m tools.train_alternate --help`.

## Testing
* Start testing by run `python -m tools.test_final`. Variable args can be found by run
`python -m tools.test_final --help`.

## Contributing Guide
You are more than welcome to add new features to this implementation or fix any potential bugs. 
Here are some topics to look at.
* MXNet features superior and robust distributed training. This implementation 
has not yet fully ultilized this power.
* New approximate end to end training is available from Faster R-CNN python 
implementation whose link can be found in Disclaimer. This implementation 
does not support this feature.
* MXNet has efficient data loading module which renders data IO irrelevant 
in performance. This implementation has not used this module.
* More object detection dataset is available online. The dataset module is designed 
as simple and scalable. Welcome to add more dataset support to this implementation.
* During inference, some operations are only conducted in cpu. Reimplement them may bring 
better performance in testing time.

## Disclaimer
This repository used code from [MXNet](https://github.com/dmlc/mxnet),
[Fast R-CNN](https://github.com/rbgirshick/fast-rcnn),
[Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn),
[caffe](https://github.com/BVLC/caffe). Training data are from
[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/),
[ImageNet](http://image-net.org/). Model comes from
[VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).

## References
1. Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu, Chiyuan Zhang, and Zheng Zhang. MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems. In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015
2. Ross Girshick. "Fast R-CNN." In Proceedings of the IEEE International Conference on Computer Vision, 2015.
3. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards real-time object detection with region proposal networks." In Advances in Neural Information Processing Systems, 2015.
4. Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. "Caffe: Convolutional architecture for fast feature embedding." In Proceedings of the ACM International Conference on Multimedia, 2014.
5. Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. "The pascal visual object classes (voc) challenge." International journal of computer vision 88, no. 2 (2010): 303-338.
6. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. "ImageNet: A large-scale hierarchical image database." In Computer Vision and Pattern Recognition, IEEE Conference on, 2009.
7. Karen Simonyan, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
