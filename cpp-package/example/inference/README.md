# MXNet C++ Package Inference Workflow Examples

## Building C++ Inference examples

The examples in this folder demonstrate the **inference** workflow. Please build the MXNet C++ Package as explained in the [README](<https://github.com/apache/incubator-mxnet/tree/master/cpp-package#building-c-package>) File before building these examples.
To build examples use following commands:

-  Release: **make all**
-  Debug: **make debug all**


## Examples demonstrating inference workflow

This directory contains following examples. In order to run the examples, ensure that the path to the MXNet shared library is added to the OS specific environment variable viz. **LD\_LIBRARY\_PATH** for Linux, Mac and Ubuntu OS and **PATH** for Windows OS.

### [inception_inference.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference/inception_inference.cpp>)

This example demonstrates image classification workflow with pre-trained models using MXNet C++ API. The command line parameters the example can accept are as shown below:

```
./inception_inference --help
Usage:
inception_inference --symbol <model symbol file in json format>
                    --params <model params file>
					--image <path to the image used for prediction
					--synset file containing labels for prediction
					[--input_shape <dimensions of input image e.g "3 224 224"]
					[--mean file containing mean image for normalizing the input image
					[--gpu] Specify this option if workflow needs to be run in gpu context
```
The model json and param file and synset files are required to run this example.  The sample command line is as follows:

```

./inception_inference --symbol "./model/Inception-BN-symbol.json" --params "./model/Inception-BN-0126.params" --synset "./model/synset.txt" --mean "./model/mean_224.nd" --image "./model/dog.jpg"
```
Alternatively, The script [unit_test_inception_inference.sh](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference/unit_test_inception_inference.sh>) downloads the pre-trained **Inception** model and a test image. The users can invoke this script as follows:

```
./unit_test_inception_inference.sh
```

### [simple_rnn.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference/simple_rnn.cpp>)
This example demonstrates how you can load a pre-trained RNN model and use it to generate an output sequence with the MXNet C++ API.
The example performs the following tasks
- Loads the pre-trained RNN model.
- Loads the dictionary file containing the word to index mapping.
- Convert the input string to vector of indices and padded to match the input data length.
- Run the forward pass and predict the output string.

The example uses a pre-trained RNN model that is trained with the dataset containing speeches given by Obama.
The model consists of :
- Embedding Layer with the size of embedding to be 650
- 3 LSTM Layers with hidden dimension size of 650 and sequence length of 35
- FullyConnected Layer
- SoftmaxOutput
The model was trained for 100 epochs.
The visual representation of the model is [here](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference/obama-speaks.pdf>).

The model files can be found here.
- [obama-speaks-symbol.json](<https://s3.amazonaws.com/mxnet-cpp/RNN_model/obama-speaks-symbol.json>)
- [obama-speaks-0100.params](<https://s3.amazonaws.com/mxnet-cpp/RNN_model/obama-speaks-0100.params>)
- [obama.dictionary.txt](<https://s3.amazonaws.com/mxnet-cpp/RNN_model/obama.dictionary.txt>) Each line of the dictionary file contains a word and a unique index for that word, separated by a space, with a total of 14293 words generated from the training dataset.
The example downloads the above files while running.

The example's command line parameters are as shown below:

```
./simple_rnn --help
Usage:
simple_rnn
[--input] Input string sequence.
[--gpu]  Specify this option if workflow needs to be run in gpu context.

./simple_rnn

or

./simple_rnn --input "Good morning. I appreciate the opportunity to speak here"
```

The example will output the sequence of 35 words as follows:
```
[waters elected Amendment Amendment Amendment Amendment retirement maximize maximize maximize acr citi sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio sophisticatio ]
```

Alternatively, user can run [unit_test_simple_rnn.sh](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference/unit_test_simple_rnn.sh>) script.
