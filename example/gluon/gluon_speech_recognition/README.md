Mxnet speech recognition gluon


This project is a CNN CTC model written on the gluon interface of mxnet. It is mainly used for speech recognition. The data set used in this project is thchs30.


MFCC feature is used in audio feature extraction. I changed the feature extraction into np. savetxt before training to facilitate loading. The code for feature extraction is audio.py and audio_utils.py, which are moved from other projects and erased.


In this project, the sequential_CNN_CTC.py file is built using the native gluon's Sequntial model, and hybridSequential_CNN_CTC.py is rewritten using HybridSequntial, mainly to improve the efficiency of the GPU.
