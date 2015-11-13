# LSTM char-rnn

Because we explicitly unroll the LSTM/RNN over time for a fixed sequence length,
it is easy to fit this model into the existing FeedForward model and re-use everything.
To get a more flexible LSTM/RNN implementation that avoids explicit unrolling and
deals with variable-length sequences, we still need to implement another model
beside the existing FeedForward.

To run this example, you will need to install two extra Julia packages: `Iterators.jl`
and `StatsBase.jl`.

## Training

This example is adapted from the
[example in Python binding](https://github.com/dmlc/mxnet/blob/master/example/rnn/char_lstm.ipynb) of
MXNet. The data `input.txt` can be downloaded [here](https://github.com/dmlc/web-data/tree/master/mxnet/tinyshakespeare).

Modify parameters in [config.jl](config.jl) and then run [train.jl](train.jl). An example output
of training looks like this:
```
...
INFO: Speed: 357.72 samples/sec
INFO: == Epoch 020 ==========
INFO: ## Training summary
INFO:                NLL = 1.4672
INFO:         perplexity = 4.3373
INFO:               time = 87.2631 seconds
INFO: ## Validation summary
INFO:                NLL = 1.6374
INFO:         perplexity = 5.1418
INFO: Saved checkpoint to 'char-lstm/checkpoints/ptb-0020.params'
INFO: Speed: 368.74 samples/sec
INFO: Speed: 361.04 samples/sec
INFO: Speed: 360.02 samples/sec
INFO: Speed: 362.34 samples/sec
INFO: Speed: 360.80 samples/sec
INFO: Speed: 362.77 samples/sec
INFO: Speed: 357.18 samples/sec
INFO: Speed: 355.30 samples/sec
INFO: Speed: 362.33 samples/sec
INFO: Speed: 359.23 samples/sec
INFO: Speed: 358.09 samples/sec
INFO: Speed: 356.89 samples/sec
INFO: Speed: 371.91 samples/sec
INFO: Speed: 372.24 samples/sec
INFO: Speed: 356.59 samples/sec
INFO: Speed: 356.64 samples/sec
INFO: Speed: 360.24 samples/sec
INFO: Speed: 360.32 samples/sec
INFO: Speed: 362.38 samples/sec
INFO: == Epoch 021 ==========
INFO: ## Training summary
INFO:                NLL = 1.4655
INFO:         perplexity = 4.3297
INFO:               time = 86.9243 seconds
INFO: ## Validation summary
INFO:                NLL = 1.6366
INFO:         perplexity = 5.1378
INFO: Saved checkpoint to 'examples/char-lstm/checkpoints/ptb-0021.params'
```

## Sampling

Run [sampler.jl](sampler.jl) to generate sample sentences from the trained model. Some example sentences are
```
## Sample 1
all have sir,
Away will fill'd in His time, I'll keep her, do not madam, if they here? Some more ha?

## Sample 2
am.

CLAUDIO:
Hone here, let her, the remedge, and I know not slept a likely, thou some soully free?

## Sample 3
arrel which noble thing
The exchnachsureding worns: I ne'er drunken Biancas, fairer, than the lawfu?

## Sample 4
augh assalu, you'ld tell me corn;
Farew. First, for me of a loved. Has thereat I knock you presents?

## Sample 5
ame the first answer.

MARIZARINIO:
Door of Angelo as her lord, shrield liken Here fellow the fool ?

## Sample 6
ad well.

CLAUDIO:
Soon him a fellows here; for her fine edge in a bogms' lord's wife.

LUCENTIO:
I?

## Sample 7
adrezilian measure.

LUCENTIO:
So, help'd you hath nes have a than dream's corn, beautio, I perchas?

## Sample 8
as eatter me;
The girlly: and no other conciolation!

BISTRUMIO:
I have be rest girl. O, that I a h?

## Sample 9
and is intend you sort:
What held her all 'clama's for maffice. Some servant.' what I say me the cu?

## Sample 10
an thoughts will said in our pleasue,
Not scanin on him that you live; believaries she.

ISABELLLLL?
```
