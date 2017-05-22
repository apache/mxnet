# Use data from S3 for training

AWS S3 is a cloud-based object storage service that allows storage and retrieval of large amounts of data at a very low cost. This makes it an attractive option to store large training datasets. MXNet is deeply integrated with S3 for this purpose.

An S3 protocol URL (like `s3://bucket-name/training-data`) can be provided as a parameter for any data iterator that takes a file path as input. For example,

```
data_iter = mx.io.ImageRecordIter(
    path_imgrec="s3://bucket-name/training-data/caltech_train.rec",
    data_shape=(3, 227, 227),
    batch_size=4,
    resize=256)
```
Following are detailed instructions on how to use data from S3 for training.

## Step 1: Build MXNet with S3 integration enabled

Follow instructions [here](http://mxnet.io/get_started/install.html) to install MXNet from source with the following additional steps to enable S3 integration.

1. Install `libcurl4-openssl-dev` and `libssl-dev` before building MXNet. These packages are required to read/write from AWS S3.
2. Append `USE_S3=1` to `config.mk` before building MXNet.
    ```
    echo "USE_S3=1" >> config.mk
    ```

## Step 2: Configure S3 authentication tokens

MXNet requires the S3 environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to be set. [Here](https://aws.amazon.com/blogs/security/wheres-my-secret-access-key/) are instructions to get the access keys from AWS console.

```
export AWS_ACCESS_KEY_ID=<your-access-key-id>
AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
```

## Step 3: Upload data to S3

There are several ways to upload data to S3. One easy way is to use the AWS command line utility. For example, the following `sync` command will recursively copy contents from a local directory to a directory in S3.

```
aws s3 sync ./training-data s3://bucket-name/training-data
```

## Step 4: Train with data from S3

Once the data is in S3, it is very straightforward to use it from MXNet. Any data iterator that can read/write data from a local drive can also read/write data from S3.

Let's modify an existing example code in MXNet repository to read data from S3 instead of local disk. [`mxnet/tests/python/train/test_conv.py`](https://github.com/dmlc/mxnet/blob/master/tests/python/train/test_conv.py) trains a convolutional network using MNIST data from local disk. We'll do the following change to read the data from S3 instead.

```
~/mxnet$ sed -i -- 's/data\//s3:\/\/bucket-name\/training-data\//g' ./tests/python/train/test_conv.py

~/mxnet$ git diff ./tests/python/train/test_conv.py
diff --git a/tests/python/train/test_conv.py b/tests/python/train/test_conv.py
index 039790e..66a60ce 100644
--- a/tests/python/train/test_conv.py
+++ b/tests/python/train/test_conv.py
@@ -39,14 +39,14 @@ def get_iters():

     batch_size = 100
     train_dataiter = mx.io.MNISTIter(
-            image="data/train-images-idx3-ubyte",
-            label="data/train-labels-idx1-ubyte",
+            image="s3://bucket-name/training-data/train-images-idx3-ubyte",
+            label="s3://bucket-name/training-data/train-labels-idx1-ubyte",
             data_shape=(1, 28, 28),
             label_name='sm_label',
             batch_size=batch_size, shuffle=True, flat=False, silent=False, seed=10)
     val_dataiter = mx.io.MNISTIter(
-            image="data/t10k-images-idx3-ubyte",
-            label="data/t10k-labels-idx1-ubyte",
+            image="s3://bucket-name/training-data/t10k-images-idx3-ubyte",
+            label="s3://bucket-name/training-data/t10k-labels-idx1-ubyte",
             data_shape=(1, 28, 28),
             label_name='sm_label',
             batch_size=batch_size, shuffle=True, flat=False, silent=False)
```

After the above change `test_conv.py` will fetch data from S3 instead of the local disk.

```
python ./tests/python/train/test_conv.py
[21:59:19] src/io/s3_filesys.cc:878: No AWS Region set, using default region us-east-1
[21:59:21] src/io/iter_mnist.cc:94: MNISTIter: load 60000 images, shuffle=1, shape=(100,1,28,28)
[21:59:21] src/io/iter_mnist.cc:94: MNISTIter: load 10000 images, shuffle=1, shape=(100,1,28,28)
INFO:root:Start training with [cpu(0)]
Start training with [cpu(0)]
INFO:root:Epoch[0] Resetting Data Iterator
Epoch[0] Resetting Data Iterator
INFO:root:Epoch[0] Time cost=11.277
Epoch[0] Time cost=11.277
INFO:root:Epoch[0] Validation-accuracy=0.955100
Epoch[0] Validation-accuracy=0.955100
INFO:root:Finish fit...
Finish fit...
INFO:root:Finish predict...
Finish predict...
INFO:root:final accuracy = 0.955100
final accuracy = 0.955100
```
