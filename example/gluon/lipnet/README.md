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

This is a Gluon implementation of [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599)

![net_structure](asset/network_structure.png)

![sample output](https://user-images.githubusercontent.com/11376047/52533982-d7227680-2d7e-11e9-9f18-c15b952faf0e.png)

## Requirements
- Python 3.6.4
- MXNet 1.3.0
- Required disk space: 35 GB
```
pip install -r requirements.txt
```

---

## The Data
- The GRID audiovisual sentence corpus (http://spandh.dcs.shef.ac.uk/gridcorpus/)
  - GRID is a large multi-talker audiovisual sentence corpus to support joint computational-behavioral studies in speech perception. In brief, the corpus consists of high-quality audio and video (facial) recordings of 1000 sentences spoken by each of 34 talkers (18 male, 16 female). Sentences are of the form "put red at G9 now". The corpus, together with transcriptions, is freely available for research use.
- Video: (normal)(480 M each)
  - Each movie has one sentence consist of 6 words.
- Align: word alignments (190 K each)
  - One align has 6 words. Each word has start time and end time. But this tutorial needs just sentence because of using ctc-loss.

---

## Pretrained model
You can train the model yourself in the following sections, you can test a pretrained model's inference, or resume training from the model checkpoint. To work with the provided pretrained model, first download it, then run one of the provided Python scripts for inference (infer.py) or training (main.py).

* Download the [pretrained model](https://github.com/soeque1/temp_files/files/2848870/epoches_81_loss_15.7157.zip)
* Try inference with the following:

```
python infer.py model_path='checkpoint/epoches_81_loss_15.7157'
```

* Resume training with the following:

```
python main.py model_path='checkpoint/epoches_81_loss_15.7157'
```

## Prepare the Data

You can prepare the data yourself, or you can download preprocessed data.

### Option 1 - Download the preprocessed data

There are two download routes provided for the preprocessed data.

#### Download and untar the data
To download tar zipped files by link, download the following files and extract in a folder called `data` in the root of this example folder. You should have the following structure:
```
/lipnet/data/align
/lipnet/data/datasets
```

* [align files](https://mxnet-public.s3.amazonaws.com/lipnet/data-archives/align.tgz)
* [datasets files](https://mxnet-public.s3.amazonaws.com/lipnet/data-archives/datasets.tgz)

#### Use AWS CLI to sync the data
To get the folders and files all unzipped with AWS CLI, can use the following command. This will provide the folder structure for you. Run this command from `/lipnet/`:

```
 aws s3 sync s3://mxnet-public/lipnet/data .
```

### Option 2 (part 1)- Download the raw dataset
- Outputs
  - The Total Movies(mp4): 16GB
  - The Total Aligns(text): 134MB
- Arguments
  - src_path : Path for videos (default='./data/mp4s/')
  - align_path : Path for aligns (default='./data/')
  - n_process : num of process (default=1)

```
cd ./utils && python download_data.py --n_process=$(nproc)
```

### Option 2 (part 2) Preprocess the raw dataset: Extracting the mouth images from a video and save it

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

#### How to run the preprocess script

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

#### Output: Data structure of the preprocessed data

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
After you have acquired the preprocessed data you are ready to train the lipnet model.

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
  - model_path : Path of pretrained model (default=None)

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
  - model_path : Path of pretrained model (default=None)

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
