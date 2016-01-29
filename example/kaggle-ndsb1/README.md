Tutorial for Kaggle NDSB-1
-----

This is an MXNet example for Kaggle Nation Data Science Bowl 1.
Test/train image data and sample submission have to be downloaded from https://www.kaggle.com/c/datasciencebowl/data. into a "data" folder.
Uncompress train.zip and test.zip folders.

#### Step 1: Generate image list
- Prepare original data, in layout like
```
--gen_img_list.py
--data/
    |
    |--train/
    |   |
    |   |--acantharia_protist/...
    |   |--.../
    |--test/...
```
- Run command ``` python gen_img_list.py --train``` to generate a the train image list
- If no extra arguments are passed it will generate the list of the train set (train.lst) and it will also automatically split the list into a train and validation set (tr.lst and va.lst in the /data directory). There is an option to use using stratified sampling and/or to give the percentage of set that is assigned to validation.
- Run command ``` python gen_img_list.py --image-folder=data/test/ --out-file=test.lst``` to generate a test image list


#### Step 2: Generate Image Record (new shape with short edge = 48)
- ```mkdir data48```
- Run command ```../../bin/im2rec data/tr.lst ./ data48/tr.rec resize=48``` to generate training data record file
- Run command ```../../bin/im2rec data/va.lst ./ data48/va.rec resize=48``` to generate validation data record file
- Run command ```../../bin/im2rec data/test.lst ./ data48/test.rec resize=48``` to generate validation data record file

#### Step 3: Train Model
- The network structure is defined in file symbol_dsb.py
- We will use find_mxnet.py and train_model.py from the image-classification example folder. Generate simbolic links to those files ```ln -s ../image-classification/find_mxnet.py .``` and ```ln -s ../image-classification/train_model.py .```
- ```mkdir models``` , if you want to save the models in that folder.
- Run ```python train_dsb.py``` to train the model, look to the help of that file to change the parameters. (See Step 4 if you want to make training curve plot)
- Sample settings would get you

2016-01-16 22:03:48,269 Node[0] Epoch[49] Train-accuracy=0.664038
2016-01-16 22:03:48,269 Node[0] Epoch[49] Time cost=25.107
2016-01-16 22:03:51,977 Node[0] Epoch[49] Validation-accuracy=0.647807
2016-01-16 22:03:51,999 Node[0] Saved checkpoint to "./models/sample_net-0-0050.params"

#### Step 4: Plot train / validation curves.
- If you want to make a training/validation vs epoch plot you should save the log of the model fit to a logfile. For example, train the model with: ```python train_dsb.py --log-file "log_tr_va" --log-dir "." ```
- Make the plot: python training_curves.py 


#### Step 5: Test predictions
- We will use epoch no. 50 to make predictions on the test set.
- Run ```python test_dsb.py``` to make predictions on the test.rec, look to the help of that file to change the parameters
- This will call submission_dsb.py function to generate the a csv file to be submitted to kaggle leaderboard, you should get around position 325.

