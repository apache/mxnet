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

## Create a Dataset Using im2rec API

RecordIO implements a file format for a sequence of records. We recommend storing images as records and packing them together. The benefits include:

* Storing images in a compact format--e.g., JPEG, for records--greatly reduces the size of the dataset on the disk.
* Packing data together allows continuous reading on the disk.
* RecordIO has a simple way to partition, simplifying distributed setting. We provide an example later.

We provide an API to convert a dataset into RecordIO format, given a list file.

Here is a related post describing the RecordIO format and how Data Iterators are built in MXnet. [example using real-world data with im2rec.py.](https://mxnet.incubator.apache.org/tutorials/basic/data.html#loading-data-using-image-iterators)

### Prerequisites

Download the data. You don't need to resize the images manually, you can use ```im2rec``` API to resize them automatically.

### Step 1. Make an Image List File

After you download the data, you need to make an image list file.  The format is:

```
integer_image_index \t label_index \t path_to_image
```

This is an example file:

```bash
95099  464.000000     n04467665_17283.JPEG
10025081        412.000000     ILSVRC2010_val_00025082.JPEG
74181   789.000000     n01915811_2739.JPEG
10035553        859.000000     ILSVRC2010_val_00035554.JPEG
10048727        929.000000     ILSVRC2010_val_00048728.JPEG
94028   924.000000     n01980166_4956.JPEG
1080682 650.000000     n11807979_571.JPEG
972457  633.000000     n07723039_1627.JPEG
7534    11.000000      n01630670_4486.JPEG
1191261 249.000000     n12407079_5106.JPEG
```

### Step 2. Create the Binary File

We will first create a transforms object and pass it as an argument to im2rec API.

```python
import os
import multiprocessing as mp

list_file = 'images.lst'
output_path = os.path.join(os.getcwd(), 'images_rec')
transformations = transforms.Compose([transforms.Resize(300),
									transforms.ToTensor(),
									transforms.Normalize(0, 1)])
im2rec(list_file, output_path, transformations=transformations,
		num_workers=mp.cpu_count() - 1, batch_size=4096,
		pack_labels=True, color=1, encoding='.jpg',
        quality=95, pass_through=False, error_limit=0)
```

Depending on the machine in which you are running and the dataset you are trying to create we choose approptiate values of `num_workers` and `batch_size`.
