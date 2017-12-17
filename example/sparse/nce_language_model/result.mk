## Dense, PTB

### medium, 84.1

```
# (original command - more dropout)
python train.py --nhid 650 --emsize 650  --lr=1  --num-gpus=1 --batch_size=32 --bptt=35 --k=60  --gpu=2  --dropout=0.2 --rescale-grad=1 --mom=0.99  --tied --use-dense --lr-decay=0.5 --wd=1e-5

2017-12-17 19:23:04,317 Namespace(Z=1, batch_size=32, bptt=35, clip=0.2, data='./data/ptb.', dropout=0.2, dummy_iter=False, emsize=650, epochs=40, gpu=7, k=60, kvstore=None, log_interval=200, lr=1.0, lr_decay=0.5, mom=0.99, nhid=650, nlayers=2, num_gpus=1, profile=False, rescale_grad=1.0, scale=1, tied=True, use_dense=True, use_full_softmax=False, use_gpu=0, wd=1e-05)

2017-12-17 19:25:05,669 Iter[0] Batch [200]  Speed: 61.91 samples/sec
2017-12-17 19:25:05,669 Iter[0] Batch [200]  loss 5.6081595, ppl 272.6419879
2017-12-17 19:26:49,969 Iter[0] Batch [400]     Speed: 61.36 samples/sec
2017-12-17 19:26:49,969 Iter[0] Batch [400]     loss 4.6637771, ppl 106.0358378
2017-12-17 19:28:35,178 Iter[0] Batch [600]  Speed: 60.83 samples/sec
2017-12-17 19:28:35,180 Iter[0] Batch [600]  loss 4.3975659, ppl 81.2528524
2017-12-17 19:30:19,709 Iter[0] Batch [800]  Speed: 61.23 samples/sec
2017-12-17 19:30:19,709 Iter[0] Batch [800]  loss 4.2612039, ppl 70.8952846
2017-12-17 19:30:37,857 Iter[0] Valid        loss 5.4466710, ppl 231.9846011
2017-12-17 19:30:41,846 Iter[0] Test         loss 5.4168177, ppl 225.1614391
...
2017-12-17 23:25:03,375 Iter[32] Valid       loss 4.4648700, ppl 86.9097330
2017-12-17 23:25:07,415 Iter[32] Test        loss 4.4324336, ppl 84.1359240
```

## Sparse, PTB

### small, 103.7

```
python train.py --nhid 200 --emsize 200  --lr=1  --num-gpus=1 --batch_size=32 --bptt=35 --k=60  --gpu=2  --dropout=0.2 --rescale-grad=1 --mom=0.9  --tied --lr-decay=0.25

2017-12-17 19:27:43,531 Namespace(Z=1, batch_size=32, bptt=35, clip=0.2, data='./data/ptb.', dropout=0.5, dummy_iter=False, emsize=200, epochs=80, gpu=2, k=60, kvstore=None, log_interval=999999, lr=1.0, lr_decay=0.25, mom=0.9, nhid=200, nlayers=2, num_gpus=1, profile=False, rescale_grad=1, scale=1, tied=True, use_dense=False, use_full_softmax=False, use_gpu=0, wd=0.0)

2017-12-17 19:30:26,213 Iter[0] Valid           loss 5.8491263, ppl 346.9311454
2017-12-17 19:30:28,680 Iter[0] Test            loss 5.8147850, ppl 335.2193113
...
...
2017-12-17 21:20:26,833 Iter[44] Valid          loss 4.6809166, ppl 107.8689021
2017-12-17 21:20:29,298 Iter[44] Test           loss 4.6413675, ppl 103.6860460
```
