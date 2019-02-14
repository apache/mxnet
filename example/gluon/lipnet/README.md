<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
--->

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

---

## The Data
- The GRID audiovisual sentence corpus (http://spandh.dcs.shef.ac.uk/gridcorpus/)
  - GRID is a large multitalker audiovisual sentence corpus to support joint computational-behavioral studies in speech perception. In brief, the corpus consists of high-quality audio and video (facial) recordings of 1000 sentences spoken by each of 34 talkers (18 male, 16 female). Sentences are of the form "put red at G9 now". The corpus, together with transcriptions, is freely available for research use.
- Video: (normal)(480 M each)
  - Each movie has one sentence consist of 6 words.
- Align: word alignments(190 K each) 
  - One align has 6 words. Each word has start time and end time. But this tutorial needs just sentence because of using ctc-loss.
 
---

## Prepare the Data
### (1) Download the data
- Outputs
  - The Total Moives(mp4): 16GB
  - The Total Aligns(text): 134MB
- Arguments
  - src_path : Path for videos (default='./data/mp4s/')
  - align_path : Path for aligns (default='./data/')
  - n_process : num of process (default=1)

```
cd ./utils && python download_data.py --n_process=$(nproc)
```

### (2) Preprocess the Data: Extracting the mouth images from a video and save it.

* Using Face Landmark Detection(http://dlib.net/)

#### Preprocess (preprocess_data.py)
*  If there is no landmark, it download automatically.  
*  Using Face Landmark Detection, It extract the mouth from a video.  

- example: 
 - video: ./data/mp4s/s2/bbbf7p.mpg
 - align(target): ./data/align/s2/bbbf7p.align  
     : 'sil bin blue by f seven please sil'


- Video to the images (75 Frames)

Frame 0            |  Frame 1 | ... | Frame 74 |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](asset/s2_bbbf7p_000.png)  |  ![](asset/s2_bbbf7p_001.png) |  ...  |  ![](asset/s2_bbbf7p_074.png)

  - Extract the mouth from images

Frame 0            |  Frame 1 | ... | Frame 74 |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](asset/mouth_000.png)  |  ![](asset/mouth_001.png) |  ...  |  ![](asset/mouth_074.png)

* Save the result images into tgt_path.  

----

### How to run

- Arguments
  - src_path : Path for videos (default='./data/mp4s/')
  - tgt_path : Path for preprocessed images (default='./data/datasets/')
  - n_process : num of process (default=1)

- Outputs
  - The Total Images(png): 19GB
- Elapsed time
  - About 54 Hours using 1 process
  - If you use the multi-processes, you can finish the number of processes faster.
    - e.g) 9 hours using 6 processes

You can run the preprocessing with just one processor, but this will take a long time (>48 hours). To use all of the available processors, use the following command: 

```
cd ./utils && python preprocess_data.py --n_process=$(nproc)
```

## Output: Data Structure

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

---

## Training

- According to [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599), four (S1, S2, S20, S22) of the 34 subjects are used for evaluation.
 The other subjects are used for training.
 
- To use the multi-gpu, it is recommended to make the batch size $(num_gpus) times larger.

  - e.g) 1-gpu and 128 batch_size > 2-gpus 256 batch_size


- arguments
  - batch_size : Define batch size (default=64)
  - epochs : Define total epochs (default=100)
  - image_path : Path for lip image files (default='./data/datasets/')
  - align_path : Path for align files (default='./data/align/')
  - dr_rate : Dropout rate(default=0.5)
  - num_gpus : Num of gpus (if num_gpus is 0, then use cpu) (default=1)
  - num_workers : Num of workers when generating data (default=0)
  - model_path : Path of pretrained model (defalut=None)
  
```
python main.py
```

---

## Test Environment
- 72 CPU cores
- 1 GPU (NVIDIA Tesla V100 SXM2 32 GB)
- 128 Batch Size

  -  It takes over 24 hours (60 epochs) to get some good results.

---

## Inference

- arguments
  - batch_size : Define batch size (default=64)
  - image_path : Path for lip image files (default='./data/datasets/')
  - align_path : Path for align files (default='./data/align/')
  - num_gpus : Num of gpus (if num_gpus is 0, then use cpu) (default=1)
  - num_workers : Num of workers when generating data (default=0)
  - data_type : 'train' or 'valid' (defalut='valid')
  - model_path : Path of pretrained model (defalut=None)
    
```
python infer.py --model_path=$(model_path)
```


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
  

