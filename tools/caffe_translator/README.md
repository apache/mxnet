# CaffeTranslator
CaffeTranslator is a source code translator that translates Caffe code into MXNet Python code. Note that this is different from the Caffe to MXNet model converted which is available [here](https://github.com/apache/incubator-mxnet/tree/master/tools/caffe_converter).

CaffeTranslator takes the training/validation prototxt (example) and solver prototxt (example) for a Caffe model as input and produces MXNet Python code (example) that builds and trains the same network specified in the prototxt.

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
