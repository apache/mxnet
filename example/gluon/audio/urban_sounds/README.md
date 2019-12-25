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

# Urban Sounds Classification in MXNet Gluon

This example provides an end-to-end pipeline for a common datahack competition - Urban Sounds Classification Example.
Below is the link to the competition:
https://datahack.analyticsvidhya.com/contest/practice-problem-urban-sound-classification/

After logging in, the data set can be downloaded.
The details of the dataset and the link to download it are given below:


## Urban Sounds Dataset:
### Description
  The dataset contains 8732 wav files which are audio samples(<= 4s)) of street sounds like engine_idling, car_horn, children_playing, dog_barking and so on.
  The task is to classify these audio samples into one of the following 10 labels:
  ```
  siren,
  street_music,
  drilling,
  dog_bark,
  children_playing,
  gun_shot,
  engine_idling,
  air_conditioner,
  jackhammer,
  car_horn
  ```

To be able to run this example:

1. `pip install -r requirements.txt`

    If you are in the directory where the requirements.txt file lies,
    this step installs the required libraries to run the example.
    The main dependency that is required is: Librosa. 
    The version used to test the example is: `0.6.2`
    For more details, refer here:
https://librosa.github.io/librosa/install.html

2. Download the dataset(train.zip, test.zip) required for this example from the location:
https://drive.google.com/drive/folders/0By0bAi7hOBAFUHVXd1JCN3MwTEU

3. Extract both the zip archives into the **current directory** - after unzipping you would get 2 new folders namely,
   **Train** and **Test** and two csv files - **train.csv**, **test.csv**

   Assuming you are in a directory *"UrbanSounds"*, after downloading and extracting train.zip, the folder structure should be:
   
   ```
        UrbanSounds        
                    - Train
                        - 0.wav, 1.wav ...
                    - train.csv
                    - train.py
                    - predict.py ...
    ```

4. Apache MXNet is installed on the machine. For instructions, go to the link: https://mxnet.apache.org/install/



For information on the current design of how the AudioFolderDataset is implemented, refer below:
https://cwiki.apache.org/confluence/display/MXNET/Gluon+-+Audio

### Usage 

For training:

- Arguments
  - train : The folder/directory that contains the audio(wav) files locally. Default = "./Train"
  - csv: The file name of the csv file that contains audio file name to label mapping. Default = "train.csv"
  - epochs : Number of epochs to train the model. Default = 30
  - batch_size : The batch size for training. Default = 32


###### To use the default arguments, use:
```
python train.py
``` 
or

###### To pass command-line arguments for training data directory, epochs, batch_size, csv file name, use :
```
python train.py --train ./Train --csv train.csv --batch_size 32 --epochs 30 
```

For prediction:

- Arguments
  - pred : The folder/directory that contains the audio(wav) files which are to be classified. Default = "./Test"


###### To use the default arguments, use:
```
python predict.py
``` 
or

###### To pass command-line arguments for test data directory, use :
```
python predict.py --pred ./Test
```