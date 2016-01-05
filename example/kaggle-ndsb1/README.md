Tutorial for Kaggle NDSB-1
-----

This is an MXNet example for Kaggle Nation Data Science Bowl 1.

In this example we ignored submission part, only show local validation result.

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
    |--sampleSubmission.csv
```
- Run command ``` python gen_img_list.py train data/sampleSubmission.csv data/train/ train.lst``` to generate a full image list
- Run command ```sed -n '1, 20000p' train.lst > tr.lst``` to generate local train list
- Run command ```sed -n '20001p, 30337p' train.lst > va.lst``` to generate local validation list


#### Step 2: Generate Image Record (new shape with short edge = 48)
- Run command ```../../bin/im2rec tr.lst ./ tr.rec resize=48``` to generate training data record file
- Run command ```../../bin/im2rec va.lst ./ va.rec resize=48``` to generate validation data record file

#### Step 3: Train Model
- Feel free to change hyper parameter in ```run_local.py```
- Run ```python run_local.py``` to train the model
- Sample code result: Train-accuracy=60.1%,  Validation-accuracy=62.1%


