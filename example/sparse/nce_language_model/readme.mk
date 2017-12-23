```

PYTHONPATH=~/nce/python python train.py --nhid 5 --emsize 5 --batch_size=128 --k=6 --dropout=0.5 --mom=0.95  --lr-decay=0.5 --optimizer=sgd --gpus=0,1,2,3,4,5,6,7 --checkpoint-dir=./checkpoint2/ --train-data=/home/ubuntu/small-gbw/training-monolingual.tokenized.shuffled/* --eval-data=/home/ubuntu/small-gbw/heldout-monolingual.tokenized.shuffled/* --vocab=./data/1b_word_vocab.txt

```
