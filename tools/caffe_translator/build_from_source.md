### Build Caffe Translator from source

#### Prerequisites:
- JDK

#### Instructions to build

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
gradle build
```

Caffe Translator will be built at `build/libs/caffe-translator-<version>.jar`
