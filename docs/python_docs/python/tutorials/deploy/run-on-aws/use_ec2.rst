.. Licensed to the Apache Software Foundation (ASF) under one
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

Run on an EC2 Instance
======================

This chapter shows, how to allocate a CPU/GPU instance in AWS and how to
setup the Deep Learning environment.

We first need ``an AWS account <https://aws.amazon.com/>``\ \_\_, and
then go the EC2 console after login in.

Then click "launch instance" to select the operation system and instance
type.

AWS offers
``Deep Learning AMIs <https://docs.aws.amazon.com/dlami/latest/devguide/options.html>``\ \_\_
that come with the latest versions of Deep Learning frameworks. The Deep
Learning AMIs provide all necessary packages and drivers and allow you
to directly start implementing and training your models. Deep Learning
AMIs use use binaries that are optimized to run on AWS instances to
accelerate model training and inference. In this tutorial we use Deep
Learning AMI (Ubuntu) Version 19.0:

We choose "p2.xlarge", which contains a single Nvidia K80 GPU. Note that
there is a large number of instance, refer to
``ec2instances.info <http://www.ec2instances.info/>``\ \_\_ for detailed
configurations and fees.

Note that we need to check the instance limits to guarantee that we can
request the resource. If running out of limits, we can request more
capacity by clicking the right link, which often takes about a single
workday to process.

On the next step we increased the disk from 8 GB to 40 GB so we have
enough space store a reasonable size dataset. For large-scale datasets,
we can "add new volume". Also you selected a very powerful GPU instance
such as "p3.8xlarge", make sure you selected "Provisioned IOPS" in the
volume type for better I/O performance.

Then we launched with other options as the default values. The last step
before launching is choosing the ssh key, you may need to generate and
store a key if you don't have one before.

After clicked "launch instances", we can check the status by clicking
the instance ID link.

Once the status is green, we can right-click and select "connect" to get
the access instruction.

With the given address, we can log into our instance:

The login screen will show a long list of available conda environments
for the different Deep Learning frameworks, CUDA driver and Python
versions. With ``conda activate`` you can easily switch into the
different environments. In the following example we switch to the MXNet
Python 3.6 environment:

Now you are ready to start developing and training MXNet models. Once
you start training, you can check the GPU status with ``nividia-smi``.
