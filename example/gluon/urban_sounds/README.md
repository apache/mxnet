# Urban Sounds classification in MXNet

Urban Sounds Dataset:
## Description
  The dataset contains 8732 wav files which are audio samples(<= 4s)) of street sounds like engine_idling, car_horn, children_playing, dog_barking and so on.
  The task is to classify these audio samples into one of the 10 labels.

To be able to run this example:

1. Download the dataset(train.zip, test.zip) required for this example from the location:
**https://drive.google.com/drive/folders/0By0bAi7hOBAFUHVXd1JCN3MwTEU**
  

2. Extract both the zip archives into the **current directory** - after unzipping you would get 2 new folders namely,\
   **Train** and **Test** and two csv files - **train.csv**, **test.csv**

3. Apache MXNet is installed on the machine. For instructions, go to the link: **https://mxnet.incubator.apache.org/install/**

4. Librosa is installed. To install, use the commands
   `pip install librosa`,
   For more details, refer here:
   **https://librosa.github.io/librosa/install.html**
