## **DeepSpeech2.mxnet: Rich Speech Recognition Example**
----------  
This example based on DeepSpeech2 of Baidu helps you to build Speech-To-Text (STT) models at scale using
- CNNs, fully connected networks, (Bi-) RNNs, (Bi-) LSTMs, and (Bi-) GRUs for network layers,
- batch-normalization and drop-outs for training efficiency,
- and a Baidu's WarpCTC for loss calculations.

In order to make your own STT models, besides, all you need is to just edit a configuration file not actual codes.

-------------------
## Motivation
This example is intended to guide people who want to making practical STT models with MXNet.
With rich functionalities and convenience explained above, you can build your own STT models with it easier than former examples.

-------------------  
## Before you start
#### Environments  
- MXNet version: 0.9.5+
- GPU memory size: 2.4GB+
- We recommend that you test a model using small network first.


#### Dependency: Soundfile
Install soundfile for audio preprocessing. If encounter errors about libsndfile, follow [this tutorial](http://www.linuxfromscratch.org/blfs/view/svn/multimedia/libsndfile.html).
<pre>
pip install soundfile
</pre>
#### Dependency: Warp CTC
Follow [this instruction](https://github.com/dmlc/mxnet/tree/master/example/warpctc) to install Baidu's WarpCTC.

-------------------  
## How it works


Let's introduce what it needs and how it works to run this example.

#### Data

With reference to ba-dls-deepspeech(https://github.com/baidu-research/ba-dls-deepspeech) from baidu, 
we use a **JSON** file(Libri_sample.json) for input data format as below.
```
{"duration": 2.9450625, "text": "and sharing her house which was near by", "key": "./Libri_sample/3830-12531-0030.wav"}
{"duration": 3.94, "text": "we were able to impart the information that we wanted", "key": "./Libri_sample/3830-12529-0005.wav"}
```
You can download 2 wav files above from https://github.com/samsungsds-rnd/deepspeech.mxnet/tree/master/Libri_sample.  
Download them and put it under the /path/to/yourproject/Libri_sample/  


To train with more wav files(LibriSpeech), follow the instruction below and change paths in data section of configuration file.  

```bash
git clone https://github.com/baidu-research/ba-dls-deepspeech
cd ba-dls-deepspeech
./download.sh
./flac_to_wav.sh
python create_desc_json.py /path/to/ba-dls-deepspeech/LibriSpeech/train-clean-100 train_corpus.json
python create_desc_json.py /path/to/ba-dls-deepspeech/LibriSpeech/dev-clean validation_corpus.json
python create_desc_json.py /path/to/ba-dls-deepspeech/LibriSpeech/test-clean test_corpus.json
```

#### Configuration File

The following is a sample configuration file for deep speech 2 example.  
  

```bash
vi default.cfg
```  

```
[common]
# method can be one of the followings - train,predict,load
mode = train
#ex: gpu0,gpu1,gpu2,gpu3
context = gpu0
# checkpoint prefix, check point will be saved under checkpoints folder with prefix
prefix = test_fc
# when mode is load or predict, model will be loaded from the file name with model_file under checkpoints
model_file = test_fc-0001
batch_size = 2
# log will be saved by the log_filename
log_filename = test.log
# checkpoint set n to save checkpoinsts after n epoch
save_checkpoint_every_n_epoch = 1

[data]
train_json = ./Libri_sample.json
test_json = ./Libri_sample.json
val_json = ./Libri_sample.json

language = en
width = 161
height = 1
channel = 1
stride = 1

[arch]
channel_num = 32
conv_layer1_filter_dim = [11, 41]
conv_layer1_stride = [2, 2]
conv_layer2_filter_dim = [11, 21]
conv_layer2_stride = [1, 2]

num_rnn_layer = 3
num_hidden_rnn_list = [1760, 1760, 1760]
num_hidden_proj = 0

num_rear_fc_layers = 0
num_hidden_rear_fc_list = []
act_type_rear_fc_list = []

#network: lstm, bilstm, gru, bigru
rnn_type = bigru
#vanilla_lstm or fc_lstm (no effect when network_type is gru, bigru)
lstm_type = fc_lstm
is_batchnorm = True

[train]
num_epoch = 70

learning_rate = 0.005
optimizer = adam
momentum = 0.9
# set to 0 to disable gradient clipping
clip_gradient = 0

initializer = Xavier
init_scale = 2
factor_type = in
weight_decay = 0.00001
# show progress every how many batches
show_every = 1
save_optimizer_states = True
normalize_target_k = 2

[load]
load_optimizer_states = False
```

- [common]: Configure common parameters for STT model
- [data]: Configure path and information of datasets
- [arch]: Configure architecture of network
- [train]: Parameters for train mode. Configure learning parameters and weight initializer
- [load]: Parameters for load mode.


## Run an example
-------------

#### Training

Let's train a STT model.
```bash
cd /path/to/your/project/
mkdir checkpoints
mkdir log
python main.py --configfile default.cfg
```

This will save checkpoints of your model at every n th epoch as described in default.cfg.  
  
#### Load  
  
You can load the checkpoint and make it keep training by change mode value into load in common section  
and provide file name of checkpoint as below.  
In below case, we assume that you trained your model at least to [1] epoch(epoch starts from 0) and
intend to use the checkpoint test_fc-0001 under checkpoints folder.

```bash
vi default.cfg
```  


```
[common]
# mode can be one of the followings - train, predict, load
mode = load
...
model_file = test_fc-0001
...
```

#### Predict

To predict, change mode value under common section from train to predict,
and point the model_file value to the specific checkpoint you want to load.
In below case, we assume that you trained your model at least to [1] epoch(epoch starts from 0) and
intend to use the checkpoint test_fc-0001 under checkpoints folder.

```bash
vi default.cfg
```

```
[common]
# mode can be one of the followings - train, predict, load
mode = predict
...
# *write the model you want to test*
model_file = test_fc-0001
...

[data]
train_json = ./Libri_sample.json
# *write the path of testset*
test_json = ./Libri_sample.json
...
```

```bash
python main.py --configfile default.cfg
```
----------

## Train your own network
-------------
To train your own network, you can pass option to use your own network file.  
In below case, we assume you have defined your own network in arch_custom.py.  
arch_custom.py must include function prepare_data to define actual data shape of network  
and arch to define symbol for your network.  
We modularized convolutional,fully connected,gru,bigru,lstm,bi-lstm layers to support batch norm and sequential input data,  
so you can easily make your own network.  

```bash
python main.py --configfile default.cfg --archfile arch_custom
```