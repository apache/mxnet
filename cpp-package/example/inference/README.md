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

### [sentiment_analysis_rnn.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference/sentiment_analysis_rnn.cpp>)
This example demonstrates how you can load a pre-trained RNN model and use it to predict the sentiment expressed in the given line of the movie review with the MXNet C++ API. The example performs the following tasks
- Loads the pre-trained RNN model.
- Loads the dictionary file containing the word to index mapping.
- Converts the input string to vector of indices that's truncated or padded to match the input data length.
- Runs the forward pass and predicts the sentiment score between 0 to 1 where 1 represents positive sentiment.

The example uses a pre-trained RNN model trained with a IMDB dataset. The RNN model was built by exercising the [GluonNLP Sentiment Analysis Tutorial](<http://gluon-nlp.mxnet.io/examples/sentiment_analysis/sentiment_analysis.html#>). The tutorial uses 'standard_lstm_lm_200' available in Gluon Model Zoo and fine tunes it for the IMDB dataset
The model consists of :
- Embedding Layer
- 2 LSTM Layers with hidden dimension size of 200
- Average pooling layer
- Sigmoid output layer
The model was trained for 10 epochs to achieve 85% test accuracy.
The visual representation of the model is [here](<http://gluon-nlp.mxnet.io/examples/sentiment_analysis/sentiment_analysis.html#Sentiment-analysis-model-with-pre-trained-language-model-encoder>).

The model files can be found here.
- [sentiment_analysis-symbol.json](< https://s3.amazonaws.com/mxnet-cpp/RNN_model/sentiment_analysis-symbol.json>)
- [sentiment_analysis-0010.params](< https://s3.amazonaws.com/mxnet-cpp/RNN_model/sentiment_analysis-0010.params>)
- [sentiment_token_to_idx.txt](<https://s3.amazonaws.com/mxnet-cpp/RNN_model/sentiment_token_to_idx.txt>) Each line of the dictionary file contains a word and a unique index for that word, separated by a space, with a total of 32787 words generated from the training dataset.
The example downloads the above files while running.

The example's command line parameters are as shown below:

```
./sentiment_analysis_rnn --help
Usage:
sentiment_analysis_rnn
--input Input movie review line.e.g. "This movie is the best." The input is trimmed or padded with 0s to match the max_num_words.
[--max_num_words]  The number of words in the sentence to be considered for sentiment analysis. Default is 10
[--gpu]  Specify this option if workflow needs to be run in gpu context

```

The following command line shows running the example with input line containing less number of words than max_num_words.
The input will be padded to match the max_num_words.

```
./sentiment_analysis_rnn --input "This movie has the great story and best acting"
```

The above command will output the sentiment score as follows:
```
The sentiment score between 0 and 1, (1 being positive)=0.910454
```

The following command line shows invoking the example with input having negative sentiment.

```
./sentiment_analysis_rnn --input "The movie is worst" --max_num_words 4
```
The above command will output the sentiment score as follows:
```
The sentiment score between 0 and 1, (1 being positive)=0.0315846
```

The following command line shows running the example with input line containing more number of words than max_num_words.
The input will be trimmed to match the max_num_words.

```
./sentiment_analysis_rnn --input "The best movie ever made in the history of cinema" --max_num_words 5
```
The above command will output the sentiment score as follows:
```
The sentiment score between 0 and 1, (1 being positive)=0.716847
```

Alternatively, you can run the [unit_test_sentiment_analysis_rnn.sh](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference/unit_test_sentiment_analysis_rnn.sh>) script.
