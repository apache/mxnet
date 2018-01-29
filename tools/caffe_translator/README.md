# Caffe Translator
Caffe Translator is a migration tool that helps developers migrate their existing Caffe code to MXNet and continue further development using MXNet. Note that this is different from the Caffe to MXNet model converter which is available [here](https://github.com/apache/incubator-mxnet/tree/master/tools/caffe_converter).

Caffe Translator takes the training/validation prototxt ([example](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet_train_test.prototxt)) and solver prototxt ([example](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet_solver.prototxt)) as input and produces MXNet Python code ([example](https://www.caffetranslator.org/examples/lenet/lenet_translated.py)) as output. The translated Python code uses MXNet Symbol and Module API to build the network, reads data from LMDB files ([example](https://www.caffetranslator.org/datasets/mnist.tar.gz)), runs training and saves the trained model using the MXNet Module API ([example](https://www.caffetranslator.org/examples/lenet/lenet_saved_model.tar.gz)).

### How to use

#### Get the translator:
Download the Caffe Translator from maven [repository](https://mvnrepository.com/artifact/org.caffetranslator/caffe-translator) or [build](build_from_source.md) from source. Java Runtime Environment (JRE) is required to run the translator.

#### Translate code:
To translate `train_val.prototxt` and `solver.prototxt` to MXNet Python code, run the following command:
```
java -jar caffe-translator-<version>.jar --training-prototxt <train_val_prototxt_path> \
    --solver <solver_prototxt_path> \
    --output-file <output_file_path>
```
Example:
```
java -jar caffe-translator-0.9.1.jar --training-prototxt lenet_train_test.prototxt \
    --solver lenet_solver.prototxt \
    --output-file translated_code.py
```

Here is the list of command line parameters accepted by the Caffe Translator:
- *training-prototxt*: specifies the path to the training/validation prototxt to be translated.
- *solver-prototxt*: specifies the path to the solver prototxt to be translated.
- *output-file*: specifies the file to write the translated output into.
- *params-file* (optional): specifies the .caffemodel file to initialize parameters from.
- *custom-data-layers* (optional): Specifies a comma-separated list of types of the custom data layers used in the prototxt. The translator will use [`CaffeDataIter`](https://mxnet.incubator.apache.org/faq/caffe.html#use-io-caffedataiter) to translate these layers to MXNet.

**Note:** Translated code uses [`CaffeDataIter`](https://mxnet.incubator.apache.org/faq/caffe.html#use-io-caffedataiter) to read from LMDB files. `CaffeDataIter` requires the number of examples in LMDB file to be specified as a parameter. You can provide this information before translation using a `#CaffeToMXNet` directive like shown below:

```
  data_param {
    source: "data/mnist/mnist_train_lmdb"
    #CaffeToMXNet num_examples: 60000
    batch_size: 64
    backend: LMDB
  }
```

#### Run the translated code:

Following prerequisites are required to run the translated code:
1. Caffe with MXNet interface ([Why?](faq.md#why_caffe) [How to build?](https://github.com/apache/incubator-mxnet/tree/master/plugin/caffe#install-caffe-with-mxnet-interface))
2. MXNet with Caffe plugin ([How to build?](https://github.com/apache/incubator-mxnet/tree/master/plugin/caffe#compile-with-caffe))
3. The dataset in LMDB format.

Once prerequisites are installed, the translated Python code can be run like any other Python code:

Example:
```
python translated_code.py
```

### What layers are supported?

Caffe Translator can currently translate the following layers:

- Accuracy and Top-k
- Batch Normalization
- Concat
- Convolution
- Data<sup>*</sup>
- Deconvolution
- Eltwise
- Inner Product (Fully Connected layer)
- Flatten
- Permute
- Pooling
- Power
- Relu
- Scale<sup>*</sup>
- SoftmaxOutput

<sup>*</sup> Uses [CaffePlugin](https://github.com/apache/incubator-mxnet/tree/master/plugin/caffe)

If you want Caffe Translator to translate a layer that is not in the above list, please create an [issue](https://github.com/apache/incubator-mxnet/issues/new).
