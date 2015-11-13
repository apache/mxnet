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
INFO: Speed: 355.18 samples/sec
INFO: == Epoch 020 ==========
INFO: ## Training summary
INFO:                NLL = 1.9670
INFO:         perplexity = 7.1494
INFO:               time = 88.0757 seconds
INFO: ## Validation summary
INFO:                NLL = 2.0452
INFO:         perplexity = 7.7307
INFO: Saved checkpoint to '/cbcl/cbcl01/chiyuan/mxnet/julia/examples/char-lstm/checkpoints/ptb-0020.params'
INFO: Speed: 366.23 samples/sec
INFO: Speed: 360.19 samples/sec
INFO: Speed: 355.77 samples/sec
INFO: Speed: 356.83 samples/sec
INFO: Speed: 354.80 samples/sec
INFO: Speed: 349.89 samples/sec
INFO: Speed: 352.00 samples/sec
INFO: Speed: 358.46 samples/sec
INFO: Speed: 356.58 samples/sec
INFO: Speed: 353.03 samples/sec
INFO: Speed: 351.98 samples/sec
INFO: Speed: 365.54 samples/sec
INFO: Speed: 359.14 samples/sec
INFO: Speed: 355.60 samples/sec
INFO: Speed: 362.44 samples/sec
INFO: Speed: 359.01 samples/sec
INFO: Speed: 357.99 samples/sec
INFO: Speed: 350.07 samples/sec
INFO: Speed: 358.03 samples/sec
INFO: == Epoch 021 ==========
INFO: ## Training summary
INFO:                NLL = 1.9698
INFO:         perplexity = 7.1695
INFO:               time = 87.9392 seconds
INFO: ## Validation summary
INFO:                NLL = 2.0458
INFO:         perplexity = 7.7353
INFO: Saved checkpoint to '/cbcl/cbcl01/chiyuan/mxnet/julia/examples/char-lstm/checkpoints/ptb-0021.params'
```

## Sampling

Run [sampler.jl](sampler.jl) to generate sample sentences from the trained model. Some example sentences are
```
...
## Sample 8
a, good. Baps,
To she tur in his God twerian: well Resice hestle, the a I here's a not as I lign?
H?

## Sample 9
ame.
What high sisiss itle by mard have of on sol I cound:
And pruch you betsts; you god eie hearry?

## Sample 10
and oar
Serens Iffall as a we of tere geling pover your nive relly lers; is here whill cheadaplee k?
```
