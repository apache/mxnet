# Developing a Character-level Language model

This tutorial shows how to train a character-level language model with a multilayer recurrent neural network (RNN) using Scala. This model takes one text file as input and trains an RNN that learns to predict the next character in the sequence. In this tutorial, you train a multilayer LSTM (Long Short-Term Memory) network that generates relevant text using Barack Obama's speech patterns.

There are many documents that explain LSTM concepts. If you aren't familiar with LSTM, refer to the following before you proceed:
- Christopher Olah's [Understanding LSTM blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Training a LSTM char-rnn in Julia to Generate Random Sentences](http://dmlc.ml/mxnet/2015/11/15/char-lstm-in-julia.html)
- [Bucketing in MXNet in Python](https://github.com/dmlc/mxnet-notebooks/blob/master/python/tutorials/char_lstm.ipynb)
- [Bucketing in MXNet](http://mxnet.io/faq/bucketing.html)

## How to Use This Tutorial

There are three ways to use this tutorial:

1) Run it by copying the provided code snippets and pasting them into the Scala command line, making the appropriate changes to the input file path.

2) Reuse the code by making changes to relevant parameters and running it from command line.

3) [Run the source code directly](https://github.com/dmlc/mxnet/tree/master/scala-package/examples/src/main/scala/ml/dmlc/mxnet/examples/rnn) by running the [provided scripts](https://github.com/dmlc/mxnet/tree/master/scala-package/examples/scripts/rnn).

To run the scripts:
- Build and train the model with the [run_train_charrnn.sh script](https://github.com/dmlc/mxnet/blob/master/scala-package/examples/scripts/rnn/run_train_charrnn.sh). Edit the script as follows:

Edit the CLASS_PATH variable in the script to include your operating system-specific folder (e.g., linux-x86_64-cpu/linux-x86_64-gpu/osx-x86_64-cpu) in the path. Run the script with the following command:

```bash

    bash run_train_charrnn.sh <which GPU card to use; -1 means CPU> <input data path> <location to save the model>

    e.g.,
    bash run_train_charrnn.sh -1 ./datas/obama.txt ./models/obama

```

- Run inference with the [run_test_charrnn.sh script](https://github.com/dmlc/mxnet/blob/master/scala-package/examples/scripts/rnn/run_test_charrnn.sh). Edit the script as follows:

Edit the CLASS_PATH variable in the script to include your operating system-specific folder (e.g., linux-x86_64-cpu/linux-x86_64-gpu/osx-x86_64-cpu) in the path. Run the script with the following command:

```bash

    bash run_test_charrnn.sh <input data path> <trained model from previous script>

    e.g.,
    bash run_test_charrnn.sh ./datas/obama.txt ./models/obama
```

In this tutorial, you will accomplish the following:

-	Build an LSTM network that learns speech patterns from Barack Obama's speeches at the character level. At each time interval, the input is a character.
-	Clean up the dataset.
-	Train a model.
-	Fit the model.
-	Build the inference model.

## Prerequisites

To complete this tutorial, you need:

- MXNet. See the instructions for your operating system in [Setup and Installation](http://mxnet.io/install/index.html)
- [Scala 2.11.8](https://www.scala-lang.org/download/2.11.8.html)
- [Maven 3](https://maven.apache.org/install.html)

## Download the Data

First, download the data, which contains Barack Obama's speeches. The data is stored in a file called obama.txt and is available on [mxnet.io](http://data.mxnet.io/data/char_lstm.zip)

To download the data which contains Barack Obama's speeches:

1) Download the dataset with the following command:

    ```bash
        wget http://data.mxnet.io/data/char_lstm.zip
    ```

2) Unzip the dataset with the following command:

    ```bash
        unzip char_lstm.zip -d char_lstm/
    ```

3) The downloaded data contains President Obama's speeches. You can have sneak peek at the dataset with the following command:

    ```bash
        head -10 obama.txt
    ```

Output:
```
        Call to Renewal Keynote Address Call to Renewal Pt 1Call to Renewal Part 2 TOPIC: Our Past, Our Future & Vision for America June
        28, 2006 Call to Renewal' Keynote Address Complete Text Good morning. I appreciate the opportunity to speak here at the Call to R
        enewal's Building a Covenant for a New America conference. I've had the opportunity to take a look at your Covenant for a New Ame
        rica. It is filled with outstanding policies and prescriptions for much of what ails this country. So I'd like to congratulate yo
        u all on the thoughtful presentations you've given so far about poverty and justice in America, and for putting fire under the fe
        et of the political leadership here in Washington.But today I'd like to talk about the connection between religion and politics a
        nd perhaps offer some thoughts about how we can sort through some of the often bitter arguments that we've been seeing over the l
        ast several years.I do so because, as you all know, we can affirm the importance of poverty in the Bible; and we can raise up and
         pass out this Covenant for a New America. We can talk to the press, and we can discuss the religious call to address poverty and
         environmental stewardship all we want, but it won't have an impact unless we tackle head-on the mutual suspicion that sometimes
```

## Prepare the Data

To preprocess the dataset, define the following utility functions:

* `readContent` - Reads data from the data file.
* `buildVocab` - Maps each character to a unique Integer ID, i.e., a build a vocabulary
* `text2Id` - Encodes each sentence with an Integer ID.

Then, use these utility functions to generate vocabulary from the input text file (obama.txt).

To prepare the data:

1) Read the dataset with the following function:

    ```scala
        scala> import scala.io.Source

        import scala.io.Source

        scala> // Read file
        scala> def readContent(path: String): String = Source.fromFile(path).mkString

        readContent: (path: String)String

    ```

2) Build a vocabulary with the following function:

    ```scala
        scala> // Build  a vocabulary of what char we have in the content
        scala> def buildVocab(path: String): Map[String, Int] = {
                val content = readContent(dataPath).split("\n")
                var idx = 1 // 0 is left for zero padding
                var theVocab = Map[String, Int]()
                for (line <- content) {
                 for (char <- line) {
                   val key = s"$char"
                   if (!theVocab.contains(key)) {
                     theVocab = theVocab + (key -> idx)
                     idx += 1
                   }
                 }
                }
                theVocab
               }

               buildVocab: (path: String)Map[String,Int]
    ```

3) To assign each character a unique numerical ID, use the following function:

    ```scala
        scala> def text2Id(sentence: String, theVocab: Map[String, Int]): Array[Int] = {
                val words = for (char <- sentence) yield theVocab(s"$char")
                words.toArray
              }

              text2Id: (sentence: String, theVocab: Map[String,Int])Array[Int]
    ```

4) Now, build a character vocabulary from the dataset (obama.txt). Change the input filepath (dataPath) to reflect your settings.   

    ```scala
        scala> // Give your system path to the "obama.txt" we have downloaded using previous steps.
        scala> val dataPath = "obama.txt"
        dataPath: String = obama.txt

        scala> val vocab = buildVocab(dataPath)

        scala> vocab.size
        res23: Int = 82
    ```


## Build a Multi-layer LSTM model

Now, create a multi-layer LSTM model.

To create the model:

1) Load the helper files (`Lstm.scala`, `BucketIo.scala` and `RnnModel.scala`).
`Lstm.scala` contains the definition of the LSTM cell. `BucketIo.scala` creates a sentence iterator. `RnnModel.scala` is used for model inference. The helper files are available on the [MXNet site](https://github.com/dmlc/mxnet/tree/master/scala-package/examples/src/main/scala/ml/dmlc/mxnetexamples/rnn).
To load them, at the Scala command prompt type:

    ```scala
        scala> :load ../../../scala-package/examples/src/main/scala/ml/dmlc/mxnet/examples/rnn/Lstm.scala
        scala> :load ../../../scala-package/examples/src/main/scala/ml/dmlc/mxnet/examples/rnn/BucketIo.scala
        scala> :load ../../../scala-package/examples/src/main/scala/ml/dmlc/mxnet/examples/rnn/RnnModel.scala
    ```

2) Set the LSTM hyperparameters as follows:

    ```scala
        scala> // We can support various input lengths.
        scala> // For this problem, we cut each input sentence to a length of 129 characters.
        scala> // So we only need a fixed length bucket length.
        scala> val buckets = Array(129)
        buckets: Array[Int] = Array(129)

        scala> // hidden unit in LSTM cell
        scala> val numHidden = 512
        numHidden: Int = 512

        scala> // The embedding dimension, which maps a char to a 256 dim vector
        scala> val numEmbed = 256
        numEmbed: Int = 256

        scala> // The number of lstm layers
        scala> val numLstmLayer = 3
        numLstmLayer: Int = 3

        scala> // The batch size for training
        scala> val batchSize = 32
        batchSize: Int = 32
    ```

3) Now, construct the LSTM network as a symbolic computation graph. Type the following to create a graph in which the model is unrolled for a fixed length explicitly in time.

    ```scala
        scala> // generate symbol for a length
        scala> def symGen(seqLen: Int): Symbol = {
            Lstm.lstmUnroll(numLstmLayer, seqLen, vocab.size + 1,
                        numHidden = numHidden, numEmbed = numEmbed,
                        numLabel = vocab.size + 1, dropout = 0.2f)
          }
        symGen: (seqLen: Int)ml.dmlc.mxnet.Symbol

        scala> // create the network symbol
        scala> val symbol = symGen(buckets(0))
        symbol: ml.dmlc.mxnet.Symbol = ml.dmlc.mxnet.Symbol@3a589eed

    ```      

4) To train the model, initialize states for the LSTM and create a data iterator, which groups the data into buckets.
Note: The BucketSentenceIter data iterator supports various length examples; however, we use only the fixed length version in this tutorial.

    ```scala

        scala> // initialize states for LSTM
        scala> val initC = for (l <- 0 until numLstmLayer) yield (s"l${l}_init_c", (batchSize, numHidden))

        initC: scala.collection.immutable.IndexedSeq[(String, (Int, Int))] = Vector((l0_init_c,(32,512)),
        (l1_init_c,(32,512)), (l2_init_c,(32,512)))

        scala> val initH = for (l <- 0 until numLstmLayer) yield (s"l${l}_init_h", (batchSize, numHidden))

        initH: scala.collection.immutable.IndexedSeq[(String, (Int, Int))] = Vector((l0_init_h,(32,512)),
        (l1_init_h,(32,512)), (l2_init_h,(32,512)))

        scala> val initStates = initC ++ initH

        initStates: scala.collection.immutable.IndexedSeq[(String, (Int, Int))] =
        Vector((l0_init_c,(32,512)), (l1_init_c,(32,512)), (l2_init_c,(32,512)), (l0_init_h,(32,512)),
        (l1_init_h,(32,512)), (l2_init_h,(32,512)))

        scala> val dataTrain = new BucketIo.BucketSentenceIter(dataPath, vocab, buckets,
                                              batchSize, initStates, seperateChar = "\n",
                                              text2Id = text2Id, readContent = readContent)

        dataTrain: BucketIo.BucketSentenceIter = non-empty iterator

    ```

5) You can set more than 100 epochs, but for this tutorial, specify 75 epochs. Each epoch can take as long as 4 minutes on a GPU. In this tutorial, you will use the [ADAM optimizer](http://mxnet.io/api/scala/docs/index.html#ml.dmlc.mxnet.optimizer.Adam):

    ```scala
        scala> import ml.dmlc.mxnet._
        import ml.dmlc.mxnet._

        scala> import ml.dmlc.mxnet.Callback.Speedometer
        import ml.dmlc.mxnet.Callback.Speedometer

        scala> import ml.dmlc.mxnet.optimizer.Adam
        import ml.dmlc.mxnet.optimizer.Adam

        scala> // and we will see result by training 75 epochs
        scala> val numEpoch = 75
        numEpoch: Int = 75

        scala> // learning rate
        scala> val learningRate = 0.001f
        learningRate: Float = 0.001

    ```

6) Define the perplexity utility function for the evaluation metric which is used to calculate the negative log-likelihood during training.

    ```scala
        scala> def perplexity(label: NDArray, pred: NDArray): Float = {
                val shape = label.shape
                val size = shape(0) * shape(1)
                val labelT = {
                  val tmp = label.toArray.grouped(shape(1)).toArray
                  val result = Array.fill[Float](size)(0f)
                  var idx = 0
                  for (i <- 0 until shape(1)) {
                    for (j <- 0 until shape(0)) {
                      result(idx) = tmp(j)(i)
                      idx += 1
                    }
                  }
                  result
                }
                var loss = 0f
                val predArray = pred.toArray.grouped(pred.shape(1)).toArray
                for (i <- 0 until pred.shape(0)) {
                  loss += -Math.log(Math.max(1e-10, predArray(i)(labelT(i).toInt)).toFloat).toFloat
                }
                loss / size
                }

        perplexity: (label: ml.dmlc.mxnet.NDArray, pred: ml.dmlc.mxnet.NDArray)Float

        scala> def doCheckpoint(prefix: String): EpochEndCallback = new EpochEndCallback {
                    override def invoke(epoch: Int, symbol: Symbol,
                                        argParams: Map[String, NDArray],
                                        auxStates: Map[String, NDArray]): Unit = {
                      Model.saveCheckpoint(prefix, epoch + 1, symbol, argParams, auxStates)
                    }
                }

        doCheckpoint: (prefix: String)ml.dmlc.mxnet.EpochEndCallback

    ```

7) Define the initializer that is required for creating a model, as follows:

    ```scala
        scala> val initializer = new Xavier(factorType = "in", magnitude = 2.34f)

        initializer: ml.dmlc.mxnet.Xavier = ml.dmlc.mxnet.Xavier@54e8f10a

    ```

8) Now, you have implemented all the supporting infrastructures for the char-lstm model. To train the model, use the standard [MXNet high-level API](http://mxnet.io/api/scala/docs/index.html#ml.dmlc.mxnet.FeedForward). You can train the model on a single GPU or CPU from multiple GPUs or CPUs by changing ```scala .setContext(Array(Context.gpu(0),Context.gpu(1),Context.gpu(2),Context.gpu(3)))``` to ```scala .setContext(Array(Context.gpu(0)))```:

    ```scala
        scala> val model = FeedForward.newBuilder(symbol)
                .setContext(Array(Context.gpu(0),Context.gpu(1),Context.gpu(2),Context.gpu(3)))
                .setNumEpoch(numEpoch)
                .setOptimizer(new Adam(learningRate = learningRate, wd = 0.00001f))
                .setInitializer(initializer)
                .setTrainData(dataTrain)
                .setEvalMetric(new CustomMetric(perplexity, name = "perplexity"))
                .setBatchEndCallback(new Speedometer(batchSize, 20))
                .setEpochEndCallback(doCheckpoint("obama"))
                .build()

        model: ml.dmlc.mxnet.FeedForward = ml.dmlc.mxnet.FeedForward@4926f6c7
    ```

Now, you have an LSTM model and you've trained it. Use this model to create the inference.

## Build the Inference Model

You can now sample sentences from the trained model. The sampler works as follows:
- Takes some fixed character set (e.g., "The United States") and feeds it into the LSTM as the starting input.
- The LSTM produces an output distribution over the vocabulary and a state in the first time step then, samples a character from the output distribution and fixes it as the second character.
- In the next time step, feeds the previously sampled character as input.
- Continues running until it has sampled enough characters. Note we are running mini-batches, so several sentences could be sampled simultaneously.

To build the inference model, define the following utility functions that help MXNet make inferences:

* `makeRevertVocab` - Reverts the key value in the dictionary for easy access to characters while predicting
* `makeInput` -  Uses a given character as input
* `cdf`, `choice` - `cdf` is a helper function for the `choice` function, which is used to create random samples
* `makeOutput` - Directs the model to use either random output or fixed output by choosing the option with the greatest probability.

    ```scala
        scala> import scala.util.Random

        scala> // helper structure for prediction
        scala> def makeRevertVocab(vocab: Map[String, Int]): Map[Int, String] = {
                  var dic = Map[Int, String]()
                  vocab.foreach { case (k, v) =>
                    dic = dic + (v -> k)
                  }
                  dic
                }

      makeRevertVocab: (vocab: Map[String,Int])Map[Int,String]

      scala> // make input from char
      scala> def makeInput(char: Char, vocab: Map[String, Int], arr: NDArray): Unit = {
              val idx = vocab(s"$char")
              val tmp = NDArray.zeros(1)
              tmp.set(idx)
              arr.set(tmp)
            }

      makeInput: (char: Char, vocab: Map[String,Int], arr: ml.dmlc.mxnet.NDArray)Unit

      scala> // helper function for random sample
      scala> def cdf(weights: Array[Float]): Array[Float] = {
                val total = weights.sum
                var result = Array[Float]()
                var cumsum = 0f
                for (w <- weights) {
                  cumsum += w
                  result = result :+ (cumsum / total)
                }
                result
              }

      cdf: (weights: Array[Float])Array[Float]

      scala> def choice(population: Array[String], weights: Array[Float]): String = {
              assert(population.length == weights.length)
              val cdfVals = cdf(weights)
              val x = Random.nextFloat()
              var idx = 0
              var found = false
              for (i <- 0 until cdfVals.length) {
                if (cdfVals(i) >= x && !found) {
                  idx = i
                  found = true
                }
              }
              population(idx)
            }

      choice: (population: Array[String], weights: Array[Float])String

      scala> // we can use random output or fixed output by choosing largest probability
      scala> def makeOutput(prob: Array[Float], vocab: Map[Int, String],
                              sample: Boolean = false, temperature: Float = 1f): String = {
                 var idx = -1
                 val char = if (sample == false) {
                   idx = ((-1f, -1) /: prob.zipWithIndex) { (max, elem) =>
                     if (max._1 < elem._1) elem else max
                   }._2
                   if (vocab.contains(idx)) vocab(idx)
                   else ""
                 } else {
                   val fixDict = Array("") ++ (1 until vocab.size + 1).map(i => vocab(i))
                   var scaleProb = prob.map(x => if (x < 1e-6) 1e-6 else if (x > 1 - 1e-6) 1 - 1e-6 else x)
                   var rescale = scaleProb.map(x => Math.exp(Math.log(x) / temperature).toFloat)
                   val sum = rescale.sum.toFloat
                   rescale = rescale.map(_ / sum)
                   choice(fixDict, rescale)
                 }
                 char
               }

      makeOutput: (prob: Array[Float], vocab: Map[Int,String], sample: Boolean, temperature: Float)String

    ```

1) Build the inference model:

    ```scala
        scala> // load from check-point
        scala> val (_, argParams, _) = Model.loadCheckpoint("obama", 75)

        scala> // build an inference model
        scala> val model = new RnnModel.LSTMInferenceModel(numLstmLayer, vocab.size + 1, \
                                   numHidden = numHidden, numEmbed = numEmbed, \
                                   numLabel = vocab.size + 1, argParams = argParams, \
                                   ctx = Context.cpu(), dropout = 0.2f)

        model: RnnModel.LSTMInferenceModel = RnnModel$LSTMInferenceModel@2f0c0319
    ```

2) Now you can generate a sequence of 1200 characters (you can select any number of characters you want) starting with "The United States" as follows:

    ```scala

        scala> val seqLength = 1200
        seqLength: Int = 1200

        scala> val inputNdarray = NDArray.zeros(1)
        inputNdarray: ml.dmlc.mxnet.NDArray = ml.dmlc.mxnet.NDArray@9c231a24

        scala> val revertVocab = makeRevertVocab(vocab)

        scala> // Feel free to change the starter sentence

        scala> var output = "The United States"
        output: String = The United States

        scala> val randomSample = true
        randomSample: Boolean = true

        scala> var newSentence = true
        newSentence: Boolean = true

        scala> val ignoreLength = output.length()
        ignoreLength: Int = 17

        scala> for (i <- 0 until seqLength) {
                if (i <= ignoreLength - 1) makeInput(output(i), vocab, inputNdarray)
                else makeInput(output.takeRight(1)(0), vocab, inputNdarray)
                val prob = model.forward(inputNdarray, newSentence)
                newSentence = false
                val nextChar = makeOutput(prob, revertVocab, randomSample)
                if (nextChar == "") newSentence = true
                if (i >= ignoreLength) output = output ++ nextChar
              }

        scala> output

        res7: String = The United States who have been blessed no companies would be proud that the challenges we face, it's not as directly untelle are in my daughters - you can afford -- life-saving march care and poor information and receiving battle against other speeces and lead its people. After champions of 2006, and because Africa in America, separate has been conferenced by children ation of discrimination, we remember all of this, succeeded in any other feelings of a palently better political process - at lliims being disability payment. All across all different mights of a more just a few global personal morality and industrialized ready to succeed.One can afford when the earliest days of a pension you can add to the system be confructive despair. They have starting in the demand for...

    ```


You can see the output generated from Obama's speeches. All of the line breaks, punctuation, and uppercase and lowercase letters were produced by the sampler (no post-processing was performed).


## Next Steps
* [Scala API](http://mxnet.io/api/scala/)
* [More Scala Examples](https://github.com/dmlc/mxnet/tree/master/scala-package/examples/)
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)
