# WaveNet with Gluon

Gluon implementation of [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)

![net_structure1](assets/net_struc1.png)
![net_structure2](assets/net_struc2.png)

## Dataset
- Deepmind WaveNet blog [link](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
   - use US English data [audio link](https://storage.googleapis.com/deepmind-media/pixie/us-english/parametric-2.wav)

## Requirements
- Python 3.6.1
- Mxnet 1.2
- tqdm
- scipy.io


## Usage

- arguments
  - batch_size : Define batch size (default=64)
  - epochs : Define the total number of epochs (default=1000)
  - mu : Define mu value for [mu-law algorithm](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm) (default=128)
  - n_residue : Define number of residue (default=24)
  - dilation_depth : Define dilation depth (default=10)
  - use_gpu : whether or not to use the GPU (default=True)
  - generation : whether or not to generate a wave file for model (default=True)

###### default setting
```
python main.py
``` 
or

###### manual setting
```
python main.py --batch_size=32 --epochs=100 ...
```
## Train progress
###### 0 epoch
![epoch0](assets/progress_epoch0.png)

###### 200 epoch
![epoch200](assets/progress_epoch200.png)

###### 400 epoch
![epoch400](assets/progress_epoches400.png)


## Results
![perf_loss](assets/loss.png)


## Samples
- ground truth [link](https://soundcloud.com/seung-hwan-jung-375239472/us-english-ground-truth)


## References
- WaveNet: A Generative Model for Raw Audio | Deepmind [blog](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
- WaveNet generation code using pytorch [link](https://gist.github.com/lirnli/4282fcdfb383bb160cacf41d8c783c70#file-pytorch-wavenet-ipynb)

