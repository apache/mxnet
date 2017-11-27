# CaffeTranslator
CaffeTranslator is a source code translator that translates Caffe code into MXNet Python code. Note that this is different from the Caffe to MXNet model converted which is available [here](https://github.com/apache/incubator-mxnet/tree/master/tools/caffe_converter).

CaffeTranslator takes the training/validation prototxt ([example](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet_train_test.prototxt)) and solver prototxt ([example](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet_solver.prototxt)) as input and produces MXNet Python code that builds and trains the same network specified in the prototxt.

### How to use
To translate `train_val.prototxt` and `solver.prototxt` to MXNet, run the following command:
```
java -jar caffetranslator-<version>.jar --training-prototxt <train_val_prototxt_path> \
    --solver <solver_prototxt_path> \
    --output-file <output_file_path>
```
Example:
```
java -jar caffetranslator-0.9.0.jar --training-prototxt lenet_train_test.prototxt \
    --solver lenet_solver.prototxt \
    --output-file translated_code.py
```

**Note:** Translated code uses [`CaffeDataIter`](https://mxnet.incubator.apache.org/how_to/caffe.html#use-io-caffedataiter) to read from LMDB files. `CaffeDataIter` requires the number of examples in LMDB file to be specified as a parameter. You can provide this information using a `#caffe2mxnet` directive like shown below:

```
  data_param {
    source: "data/mnist/mnist_train_lmdb"
    #caffe2mxnet num_examples: 60000
    batch_size: 64
    backend: LMDB
  }
```

### Prerequisites
**To translate code:**
1. JDK

**To run the translated code:**
1. Caffe with MXNet interface ([Why?](faq.md#why_caffe) [How to build?](https://github.com/apache/incubator-mxnet/tree/master/plugin/caffe#install-caffe-with-mxnet-interface))
2. MXNet with Caffe plugin ([How to build?](https://github.com/apache/incubator-mxnet/tree/master/plugin/caffe#compile-with-caffe))

### Build
Step 1: Clone the code:
```
git clone https://github.com/apache/incubator-mxnet.git mxnet
```
Step 2: CD to CaffeTranslator directory
```
cd mxnet/tools/caffe_translator/
```
Step 3: Build
```
./gradlew build
```
Step 4: Install
```
./gradlew installDist
```
Step 5: Set PATH
```
PATH=$PWD/build/install/caffetranslator/bin/:$PATH
```

### Run
```
caffetranslator --training-prototxt <train_val_prototxt_path> --solver <solver_prototxt_path> --output-file <output_file_path>
```
