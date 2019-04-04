<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

**deepSpeech.mxnet: Rich Speech Example**
=========================================
  
This example based on [DeepSpeech2 of Baidu](https://arxiv.org/abs/1512.02595) helps you to build Speech-To-Text (STT) models at scale using
- CNNs, fully connected networks, (Bi-) RNNs, (Bi-) LSTMs, and (Bi-) GRUs for network layers,
- batch-normalization and drop-outs for training efficiency,
- and a Warp CTC for loss calculations.

In order to make your own STT models, besides, all you need is to just edit a configuration file not actual codes.


* * *
## **Motivation**
This example is intended to guide people who want to making practical STT models with MXNet.
With rich functionalities and convenience explained above, you can build your own speech recognition models with it easier than former examples.


* * *
## **Environments**
- MXNet version: 0.9.5+
- GPU memory size: 2.4GB+
- Install mxboard for logging
<pre>
<code>pip install mxboard</code>
</pre>  

- [SoundFile](https://pypi.python.org/pypi/SoundFile/0.8.1) for audio preprocessing (If encounter errors about libsndfile, follow [this tutorial](http://www.linuxfromscratch.org/blfs/view/svn/multimedia/libsndfile.html).)
<pre>
<code>pip install soundfile</code>
</pre>
- Warp CTC: Follow [this instruction](https://github.com/baidu-research/warp-ctc) to compile Baidu's Warp CTC. (Note: If you are using V100, make sure to use this [fix](https://github.com/baidu-research/warp-ctc/pull/118))
- You need to compile MXNet with WarpCTC, follow the instructions [here](https://github.com/apache/incubator-mxnet/tree/master/example/ctc)
- You might need to set `LD_LIBRARY_PATH` to the right path if MXNet fails to find your `libwarpctc.so`
- **We strongly recommend that you first test a model of small networks.**


* * *
## **How it works**
### **Preparing data**
Input data are described in a JSON file **Libri_sample.json** as followed.
<pre>
<code>{"duration": 2.9450625, "text": "and sharing her house which was near by", "key": "./Libri_sample/3830-12531-0030.wav"}
{"duration": 3.94, "text": "we were able to impart the information that we wanted", "key": "./Libri_sample/3830-12529-0005.wav"}</code>
</pre>
You can download two wave files above from [this](https://github.com/samsungsds-rnd/deepspeech.mxnet/tree/master/Libri_sample). Put them under /path/to/yourproject/Libri_sample/.  


### **Setting the configuration file**
**[Notice]** The configuration file "default.cfg" included describes DeepSpeech2 with slight changes. You can test the original DeepSpeech2("deepspeech.cfg") with a few line changes to the cfg file:  
<pre><code>
[common]
...
learning_rate = 0.0003
# constant learning rate annealing by factor
learning_rate_annealing = 1.1
optimizer = sgd
...
is_bi_graphemes = True
...
[arch]
...
num_rnn_layer = 7
num_hidden_rnn_list = [1760, 1760, 1760, 1760, 1760, 1760, 1760]
num_hidden_proj = 0
num_rear_fc_layers = 1
num_hidden_rear_fc_list = [1760]
act_type_rear_fc_list = ["relu"]
...
[train]
...
learning_rate = 0.0003
# constant learning rate annealing by factor
learning_rate_annealing = 1.1
optimizer = sgd
...
</code></pre>


* * *
## **Run the example**
### **Train**
<pre><code>cd /path/to/your/project/
mkdir checkpoints
mkdir log
python main.py --configfile default.cfg</code></pre>
Checkpoints of the model will be saved at every n-th epoch.
  
### **Load**
You can (re-) train (saved) models by loading checkpoints (starting from 0). For this, you need to modify only two lines of the file "default.cfg".
<pre><code>...
[common]
# mode can be one of the followings - train, predict, load
mode = load
...
model_file = 'file name of your model saved'
...</code></pre>


### **Predict**
You can predict (or test) audios by specifying the mode, model, and test data in the file "default.cfg".
<pre><code>...
[common]
# mode can be one of the followings - train, predict, load
mode = predict
...
model_file = 'file name of your model to be tested'
...
[data]
...
test_json = 'a json file described test audios'
...</code></pre>
<br />
Run the following line after all modification explained above.  
<pre><code>python main.py --configfile default.cfg</code></pre>


* * *
## **Train and test your own models**

Train and test your own models by preparing two files.
1) A new configuration file, i.e., custom.cfg, corresponding to the file 'default.cfg'.
The new file should specify the items below the '[arch]' section of the original file. 
2) A new implementation file, i.e., arch_custom.py, corresponding to the file 'arch_deepspeech.py'.
The new file should implement two functions, prepare_data() and arch(), for building networks described in the new configuration file.

Run the following line after preparing the files.   
<pre><code>python main.py --configfile custom.cfg --archfile arch_custom</pre></code>

***
## **Further more**
You can prepare full LibriSpeech dataset by following the instruction on https://github.com/baidu-research/ba-dls-deepspeech  
**Change flac_to_wav.sh script of baidu to flac_to_wav.sh in repository to avoid bug**
```bash
git clone https://github.com/baidu-research/ba-dls-deepspeech
cd ba-dls-deepspeech
./download.sh
cp -f /path/to/example/flac_to_wav.sh ./
./flac_to_wav.sh
python create_desc_json.py /path/to/ba-dls-deepspeech/LibriSpeech/train-clean-100 train_corpus.json
python create_desc_json.py /path/to/ba-dls-deepspeech/LibriSpeech/dev-clean validation_corpus.json
python create_desc_json.py /path/to/ba-dls-deepspeech/LibriSpeech/test-clean test_corpus.json
```
