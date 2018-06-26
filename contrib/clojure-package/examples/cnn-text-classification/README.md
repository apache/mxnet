# cnn-text-classification

An example of text classification using CNN

To use you must download the MR polarity dataset and put it in the path specified in the mr-dataset-path
The dataset can be obtained here: [https://github.com/yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence). The two files `rt-polarity.neg`
and `rt-polarity.pos` must be put in a directory. For example, `data/mr-data/rt-polarity.neg`.

You also must download the glove word embeddings. The suggested one to use is the smaller 50 dimension one
`glove.6B.50d.txt` which is contained in the download file here [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

## Usage

You can run through the repl with
`(train-convnet {:embedding-size 50 :batch-size 100 :test-size 100 :num-epoch 10 :max-examples 1000})`

or
`JVM_OPTS="Xmx1g" lein run` (cpu)

You can control the devices you run on by doing:

`lein run :cpu 2` - This will run on 2 cpu devices
`lein run :gpu 1` - This will run on 1 gpu device
`lein run :gpu 2` - This will run on 2 gpu devices


The max-examples only loads 1000 each of the dataset to keep the time and memory down. To run all the examples, 
change the main to be (train-convnet {:embedding-size 50 :batch-size 100 :test-size 1000 :num-epoch 10)

and then run

- `lein uberjar`
- `java -Xms1024m -Xmx2048m -jar target/cnn-text-classification-0.1.0-SNAPSHOT-standalone.jar`
