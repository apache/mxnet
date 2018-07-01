# Large-Scale Language Model
This example implements the baseline model in
[Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410) on the
[Google 1-Billion Word](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) (GBW) dataset.

This example reaches 48.0 test perplexity after 6 training epochs on a 1-layer, 2048-unit, 512-projection LSTM Language Model.
It reaches 44.2 test perplexity after 35 epochs of training.

The main differences with the original implementation include:
* Synchronized gradient updates instead of asynchronized updates

Each epoch for training (excluding time for evaluation on test set) takes around 80 minutes on a p3.8xlarge instance, which comes with 4 Volta V100 GPUs.

# Setup dataset and build sampler
1. Download 1-Billion Word Dataset: [Link](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz)
2. Download pre-processed vocabulary file which maps tokens into ids.
3. Build sampler with cython by running `make` in the current directory. If you do not have cython installed, run `pip install cython`

# Run the Script
```
usage: train.py [-h] [--data DATA] [--test TEST] [--vocab VOCAB]
                [--emsize EMSIZE] [--nhid NHID] [--num-proj NUM_PROJ]
                [--nlayers NLAYERS] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [--dropout DROPOUT] [--eps EPS]
                [--bptt BPTT] [--k K] [--gpus GPUS]
                [--log-interval LOG_INTERVAL] [--seed SEED]
                [--checkpoint-dir CHECKPOINT_DIR] [--lr LR] [--clip CLIP]
                [--rescale-embed RESCALE_EMBED]

Language Model on GBW

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the training data
  --test TEST           location of the test data
  --vocab VOCAB         location of the corpus vocabulary file
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --num-proj NUM_PROJ   number of projection units per layer
  --nlayers NLAYERS     number of LSTM layers
  --epochs EPOCHS       number of epoch for training
  --batch-size BATCH_SIZE
                        batch size per gpu
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --eps EPS             epsilon for adagrad
  --bptt BPTT           sequence length
  --k K                 number of noise samples for estimation
  --gpus GPUS           list of gpus to run, e.g. 0 or 0,2,5. empty means
                        using gpu(0).
  --log-interval LOG_INTERVAL
                        report interval
  --seed SEED           random seed
  --checkpoint-dir CHECKPOINT_DIR
                        dir for checkpoint
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping by global norm.
  --rescale-embed RESCALE_EMBED
                        scale factor for the gradients of the embedding layer
```

To reproduce the result, run
```
train.py --gpus=0,1,2,3 --clip=10 --lr=0.2 --dropout=0.1 --eps=1 --rescale-embed=256
--test=/path/to/heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050
--data=/path/to/training-monolingual.tokenized.shuffled/*
```
