# LipNet: End-to-End Sentence-level Lipreading

---

Gluon implementation of [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599)

![net_structure](asset/network_structure.png)

## Requirements
- Python 3.6.4
- MXnet 1.3.0
- The Required Disk Space: 35Gb
```
pip install -r requirements.txt
```

## Test Environment
- 4 CPU cores
- 1 GPU (Tesla K80 12GB)


## The Data
- The GRID audiovisual sentence corpus (http://spandh.dcs.shef.ac.uk/gridcorpus/)
  - GRID is a large multitalker audiovisual sentence corpus to support joint computational-behavioral studies in speech perception. In brief, the corpus consists of high-quality audio and video (facial) recordings of 1000 sentences spoken by each of 34 talkers (18 male, 16 female). Sentences are of the form "put red at G9 now". The corpus, together with transcriptions, is freely available for research use.
- Video: (normal)(480 M each)
  - Each movie has one sentence consist of 6 words.
- Align: word alignments(190 K each) 
  - One align has 6 words. Each word has start time and end time. But this tutorial needs just sentence because of using ctc-loss.
  
## Prepare the Data
### Download the data
- Outputs
  - The Total Moives(mp4): 16GB
  - The Total Aligns(text): 134MB
- Arguments
  - src_path : Path for videos (default='./data/mp4s/')
  - align_path : Path for aligns (default='./data/')
  - n_process : num of process (default=1)

```
cd ./utils && python download_data.py --n_process $(nproc)
```

### Preprocess the Data: Extracting the mouth images from a video and save it.
- Outputs
  - The Total Images(png): 19GB
- Elapsed time
  - About 54 Hours using 1 process
  - If you use the multi-processes, you can finish the number of processes faster.
    - e.g) 9 hours using 6 processes
- Arguments
  - src_path : Path for videos (default='./data/mp4s/')
  - tgt_path : Path for preprocessed images (default='./data/datasets/')
  - n_process : num of process (default=1)

You can run the preprocessing with just one processor, but this will take a long time (>48 hours). To use all of the available processors, use the following command: 

```
cd ./utils && python preprocess_data.py --n_process $(nproc)
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
  - batch_size : Define batch size (default=64)
  - epochs : Define total epochs (default=100)
  - image_path : Path for lip image files (default='./data/datasets/')
  - align_path : Path for align files (default='./data/align/')
  - dr_rate : Dropout rate(default=0.5)
  - num_gpus : Num of gpus (if num_gpus is 0, then use cpu) (default=1)
  - num_workers : Num of workers when generating data (default=0)
  
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
  

