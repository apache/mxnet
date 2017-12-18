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

### small, 100.8

```
python train.py --nhid 200 --emsize 200  --lr=0.001  --num-gpus=1 --batch_size=32 --bptt=35 --k=60  --gpu=2  --dropout=0.2 --rescale-grad=1 --mom=0.9  --tied --lr-decay=0.25 --optimizer=adam
...
2017-12-17 23:49:10,529 Iter[0] Batch [200]  Speed: 183.71 samples/sec
2017-12-17 23:49:10,529 Iter[0] Batch [200]  loss 5.4557271, ppl 234.0950136
2017-12-17 23:49:45,053 Iter[0] Batch [400]     Speed: 185.38 samples/sec
2017-12-17 23:49:45,053 Iter[0] Batch [400]     loss 4.9596645, ppl 142.5459664
2017-12-17 23:50:19,420 Iter[0] Batch [600]     Speed: 186.22 samples/sec
2017-12-17 23:50:19,420 Iter[0] Batch [600]     loss 4.6911378, ppl 108.9771076
2017-12-17 23:50:54,095 Iter[0] Batch [800]  Speed: 184.57 samples/sec
2017-12-17 23:50:54,095 Iter[0] Batch [800]  loss 4.5524864, ppl 94.8679994
2017-12-17 23:51:01,126 Iter[0] Valid           loss 5.7913307, ppl 327.4484573
2017-12-17 23:51:03,575 Iter[0] Test            loss 5.7451951, ppl 312.6846229
...
2017-12-18 01:26:52,263 Iter[39] Valid          loss 4.6713136, ppl 106.8379974
2017-12-18 01:26:54,714 Iter[39] Test           loss 4.6128478, ppl 100.7707179
```

### medium, 85.4

```
python train.py --nhid 650 --emsize 650  --lr=0.001  --num-gpus=1 --batch_size=32 --bptt=35 --k=60  --gpu=2  --dropout=0.5 --rescale-grad=1 --mom=0.9  --tied --lr-decay=0.25 --optimizer=adam --beta1=0.9

2017-12-18 01:37:12,134 Namespace(Z=1, batch_size=32, beta1=0.9, bptt=35, clip=0.2, data='./data/ptb.', dropout=0.5, dummy_iter=False, emsize=650, epochs=80, gpu=4, k=60, kvstore=None, log_interval=999999, lr=0.001, lr_decay=0.25, mom=0.9, nhid=650, nlayers=2, num_gpus=1, optimizer='adam', profile=False, rescale_grad=1, scale=1, tied=True, use_dense=False, use_full_softmax=False, use_gpu=0, wd=0.0)

2017-12-18 01:45:47,279 Iter[0] Valid           loss 5.3994027, ppl 221.2742098
2017-12-18 01:45:51,965 Iter[0] Test            loss 5.3544731, ppl 211.5524713
2017-12-18 01:54:10,130 Iter[1] Valid           loss 5.1103995, ppl 165.7365554
2017-12-18 01:54:14,891 Iter[1] Test            loss 5.0659444, ppl 158.5300903
...
2017-12-18 05:48:45,774 Iter[29] Valid          loss 4.5050545, ppl 90.4732749
2017-12-18 05:48:50,254 Iter[29] Test           loss 4.4478014, ppl 85.4388916
```
