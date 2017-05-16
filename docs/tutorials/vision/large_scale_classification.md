# Large Scale Image Classification

Training a neural network with a large number of images present several challenges. Even with the latest GPUs it is not possible to train large networks using large amount of images in reasonable time using a single GPU. This problem can be somewhat mitigated by using multiple GPUs in a single machines. But there is limits on how many GPUs can be attached to one machine (typically 8 or 16). This tutorial explains how to train large networks with terabytes of data using multiple machines each using multiple GPUs.

## Preprocessing

### Disk space
First step in training with large data is downloading the data and preprocessing it. For this tutorial, we will be using the full imagenet dataset. Note that, at least 2 TB of disk space is required to download and preprocess this data. It is strongly recommended to use SSD instead of HDD. SSD is much better at dealing with large number of small files like images. After the preprocessing is done and images are packed into recordio files, HDD should be fine for training.

For this tutorial, we will use AWS storage instance for data preprocessing. The storage instance `i3.4xlarge` has 3.8 TB of disk space across two NVMe SSD disks. We will use software RAID to combine them into one disk and mount it at `~/data`.

```
sudo mdadm --create --verbose /dev/md0 --level=stripe --raid-devices=2 \
	/dev/nvme0n1 /dev/nvme1n1
sudo mkfs /dev/md0
sudo mkdir ~/data
sudo mount /dev/md0 ~/data
sudo chown ${whoami} ~/data
```

We now have sufficient disk space to download and preprocess the data. 

### Download imagenet

For this tutorial, we will be using the full imagenet dataset which can be downloaded from http://www.image-net.org/download-images. `fall11_whole.tar` contains all the images. This file is 1.2 TB in size and could take a long time to download.

After downloading, untar the file.
```
export ROOT=full
mkdir $ROOT
tar -xvf fall11_whole.tar -C $ROOT
```

That should give you a collection of tar files. Each tar file represents a category and contains all images that belong to that category. We can unzip each tar file and copy the images into a folder named after the name of the tar file.

```
for i in $ROOT/*.tar; do j=${i%.*}; echo $j;  mkdir -p $j; tar -xf $i -C $j; done
rm $ROOT/*.tar

ls $ROOT | head
n00004475
n00005787
n00006024
n00006484
n00007846
n00015388
n00017222
n00021265
n00021939
n00120010
```

### Remove uncommon classes for transfer learning (optional)
A common reason to train a network on Imagenet data is to then use it for transfer learning (including feature extraction or fine-tuning other models). According to this study, classes with too few images don’t help in transfer learning. So, we could remove classes with fewer than a certain number of images. The following code will remove classes with less than 500 images.

```
BAK=${ROOT}_filtered
mkdir -p ${BAK}
for c in ${ROOT}/n*; do
    count=`ls $c/*.JPEG | wc -l`
    if [ "$count" -gt "500" ]; then
        echo "keep $c, count = $count"
    else
        echo "remove $c, $count"
        mv $c ${BAK}/
    fi
done
```

### Generate a validation set
To ensure we don’t overfit the data, we will create a validation set separate from the training set, monitor the loss on the validation set frequently. We create the validation set by picking fifty random images from each class.

```
VAL_ROOT=${ROOT}_val
mkdir -p ${VAL_ROOT}
for i in ${ROOT}/n*; do
    c=`basename $i`
    echo $c
    mkdir -p ${VAL_ROOT}/$c
    for j in `ls $i/*.JPEG | shuf | head -n 50`; do
        mv $j ${VAL_ROOT}/$c/
    done
done
```

### Pack images into record files
While MXNet can read image files directly, it is recommended to pack the image files into a recordio file for increased performance. MXNet provides a tool (tools/im2rec.py) to do this. To use this tool, MXNet and OpenCV’s python module needs to be installed in the system. OpenCV’s python module can be installed on Ubuntu using the command `sudo apt-get install python-opencv`.

Set the environment variable MXNET to point to the MXNet installation directory and NAME to the name of the dataset. We assume MXNet is installed at ~/mxnet

```
MXNET=~/mxnet
NAME=full_imagenet_500_filtered
```

To create the recordIO files, we first create a list of images we want in the recordIO files and then use im2rec to pack images in the list into recordIO files. We create this list in `train_meta`. Training data is around 1T, we split it into 8 parts, which each part roughly 100 GB.

```
mkdir -p train_meta
python ${MXNET}/tools/im2rec.py --list True --chunks 8 --recursive True \
train_meta/${NAME} ${ROOT}
```

We then resize the images such that the short edge is 480 px and pack the images into recordIO files. Since most of the work is disk I/O, we use multiple (16) threads to get the work done faster.

```
python ${MXNET}/tools/im2rec.py --resize 480 --quality 90 \
--num-thread 16 train_meta/${NAME} ${ROOT}
```

Once done, we move the rec files into a folder named train.

```
mkdir -p train
mv train_meta/*.rec train/
```

We do similar preprocessing for the validation set

```
mkdir -p val_meta
python ${MXNET}/tools/im2rec.py --list True --recursive True \
val_meta/${NAME} ${VAL_ROOT}
python ${MXNET}/tools/im2rec.py --resize 480 --quality 90 \
--num-thread 16 val_meta/${NAME} ${VAL_ROOT}
mkdir -p val
mv val_meta/*.rec val/
```

We now have all training and validation images in recordIO format in `train` and `val` directories respectively. We can now use these `.rec` files for training.
