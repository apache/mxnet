# LipNet: End-to-End Sentence-level Lipreading

---

Gluon inplementation of [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599)

![net_structure](asset/network_structure.png)

## Requirements
- Python 3.6.4
- MXnet 1.3.0


## Test Environment
- 4 CPU cores
- 1 GPU (Tesla K80 12GB)


## The Data
- The GRID audiovisual sentence corpus (http://spandh.dcs.shef.ac.uk/gridcorpus/)
- Video: (normal)(480 M each)
- Align: word alignments(190 K each) 

## Prepare the Data
### Download the data
- arguments
  - src_path : Path for videos (default='./data/mp4s/')
  - align_path : Path for aligns (default='./data/align/')
  - n_process : num of process (default=1)

```
cd ./utils && python download_data.py
```

### Preprocess the Data: Extracting the mouth images from a video and save it.
- arguments
  - src_path : Path for videos (default='./data/mp4s/')
  - tgt_path : Path for preprocessed images (default='./data/datasets/')
  - n_process : num of process (default=1)

```
cd ./utils && python preprocess_data.py
```

## Data Structure

```
The training data folder should look like : 
<train_data_root>
                |--datasets
                        |--s1
                           |--bbir7s
                               |--mouth_000.png
                               |--mouth_001.png
                                   ...
                           |--bgaa8p
                               |--mouth_000.png
                               |--mouth_001.png
                                  ...
                        |--s2
                            ...
                 |--align
                         |--bw1d8a.align
                         |--bggzzs.align
                             ...

```


## Training

- arguments
  - batch_size : Define batch size (defualt=64)
  - epochs : Define total epochs (default=100)
  - image_path : Path for lip image files (default='./data/datasets/')
  - align_path : Path for align files (default='./data/align/')
  - dr_rate : dropout rate(default=0.5)
  - use_gpu : Use gpu (default=True)
  - num_workers : num of workers when generating data (default=2)
  
```
python main.py
```

## Results
```
[Target]
['lay green with a zero again',
 'bin blue with r nine please',
 'set blue with e five again',
 'bin green by t seven soon',
 'lay red at d five now',
 'bin green in x eight now',
 'bin blue with e one now',
 'lay red at j nine now']
 ```
 
 ```
[Pred]
['lay green with s zero again',
 'bin blue with r nine please',
 'set blue with e five again',
 'bin green by t seven soon',
 'lay red at c five now',
 'bin green in x eight now',
 'bin blue with m one now',
 'lay red at j nine now']
 ```
  

