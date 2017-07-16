# Word-level language modeling RNN

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on Penn Treebank (PTB) language modeling benchmark.

The model obtains the state-of-the-art result on PTB using LSTM, getting a test perplexity of ~72.

The following techniques have been adopted for SOTA results: 
- [LSTM for LM](https://arxiv.org/pdf/1409.2329.pdf)
- [Weight tying](https://arxiv.org/abs/1608.05859) between word vectors and softmax output embeddings

## Data

The PTB data is the processed version from [(Mikolov et al, 2010)](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf):

```bash
python data.py
```

## Usage

Example runs and the results:

```
python train.py --cuda --tied --nhid 650 --emsize 650 --dropout 0.5        # Test ppl of 75.3
python train.py --cuda --tied --nhid 1500 --emsize 1500 --dropout 0.65      # Test ppl of 72.0
```

<br>

`python train.py --help` gives the following arguments:
```
Optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --model MODEL      type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch_size N     batch size
  --bptt BPTT        sequence length
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --tied             tie the word embedding and softmax weights
  --cuda             Whether to use gpu
  --log-interval N   report interval
  --save SAVE        path to save the final model
```
