### To run the this example,
###
### 1.
### Get Inseption-BN model first, from here
###     https://github.com/dmlc/mxnet-model-gallery
###
### 2.
### Then Prepare 2 pictures, 1.jpg 2.jpg to extract

make
./prepare_data_with_opencv
LD_LIBRARY_PATH=../../lib/linux ./feature_extract
