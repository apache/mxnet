# Word-level language modeling RNN

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on WikiText-2 language modeling benchmark.

The model obtains ~107 ppl in WikiText-2 using LSTM.

The following techniques have been adopted for SOTA results:
- [LSTM for LM](https://arxiv.org/pdf/1409.2329.pdf)
- [Weight tying](https://arxiv.org/abs/1608.05859) between word vectors and softmax output embeddings

## Data

### Wiki Text

The wikitext-2 data is from [(The wikitext long term dependency language modeling dataset)](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/). The training script automatically loads the dataset into `$PWD/data`.


## Usage

Example runs and the results:

```
python train.py --cuda --tied --nhid 200 --emsize 200 --epochs 20  --dropout 0.2        # Test ppl of 107.49
```
```
python train.py --cuda --tied --nhid 650 --emsize 650 --epochs 40  --dropout 0.5        # Test ppl of 91.51
```
```
python train.py --cuda --tied --nhid 1500 --emsize 1500 --epochs 60  --dropout 0.65     # Test ppl of 88.42
```


<br>

`python train.py --help` gives the following arguments:
```
usage: train.py [-h] [--model MODEL] [--emsize EMSIZE] [--nhid NHID]
                [--nlayers NLAYERS] [--lr LR] [--clip CLIP] [--epochs EPOCHS]
                [--batch_size N] [--bptt BPTT] [--dropout DROPOUT] [--tied]
                [--cuda] [--log-interval N] [--save SAVE] [--gctype GCTYPE]
                [--gcthreshold GCTHRESHOLD]

MXNet Autograd RNN/LSTM Language Model on Wikitext-2.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --cuda                Whether to use gpu
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --gctype GCTYPE       type of gradient compression to use, takes `2bit` or
                        `none` for now.
  --gcthreshold GCTHRESHOLD
                        threshold for 2bit gradient compression
```
