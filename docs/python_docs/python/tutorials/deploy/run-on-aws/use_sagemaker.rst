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

Run on Amazon SageMaker
-----------------------

This chapter will give a high level overview about Amazon SageMaker,
in-depth tutorials can be found on the `Sagemaker
website <https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html>`__.

SageMaker offers Jupyter notebooks and supports MXNet out-of-the box.
You can run your notebooks on CPU instances and as such profit from the
free tier. However, more powerful CPU instances or GPU instances are
charged by time. Within this notebook you can `fetch, explore and
prepare training
data <https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-notebooks-instances.html>`__.

::

    import mxnet as mx
    import sagemaker
    mx.test_utils.get_cifar10() # Downloads Cifar-10 dataset to ./data
    sagemaker_session = sagemaker.Session()
    inputs = sagemaker_session.upload_data(path='data/cifar',
                                           key_prefix='data/cifar10')

Once the data is ready, you can easily launch training via the SageMaker
SDK. So there is no need to manually configure and log into EC2
instances. You can either bring your own model or use SageMaker's
`built-in
algorithms <https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html>`__
that are tailored to specific use cases such as computer vision, NLP
etc. SageMaker encapsulates the process of training into the class
``Estimator`` and we can now start the training on the local notebook
instance:

::

    from sagemaker.mxnet import MXNet as MXNetEstimator
    estimator = MXNetEstimator(entry_point='train.py', 
                               role=sagemaker.get_execution_role(),
                               train_instance_count=1, 
                               train_instance_type='local',
                               hyperparameters={'batch_size': 1024, 
                                                'epochs': 30})
    estimator.fit(inputs)

If you require a more powerful platform for training, then you only need
to change the ``train_instance_type``. Once you call fit, SageMaker will
automatically create the required EC2 instances, train your model within
a Docker container and then immediately shutdown these instances.
``Fit()`` requires an entry point (here ``train.py``) that describes the
model and training loop. This script needs to provides certain
functions, that will be automatically called by SageMaker once you train
and deploy the model. More information about the entry point script can
be found
`here <https://docs.aws.amazon.com/sagemaker/latest/dg/mxnet-training-inference-code-template.html>`__.
When the model is ready for deployment you can use `SageMaker's hosting
services <https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html>`__
that create an HTTPS endpoint where model inference is provided.

::

    predictor = estimator.deploy(initial_instance_count=1,
                                 instance_type='ml.m4.xlarge')

The following links show more advanced uses cases in SageMaker: -
`Distributed training on multiple
machines <https://medium.com/apache-mxnet/94-accuracy-on-cifar-10-in-10-minutes-with-amazon-sagemaker-754e441d01d7>`__
- `Hyperparameter Tuning
Jobs <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex.html>`__
- `Optimize a model with SageMaker
Neo <https://docs.aws.amazon.com/sagemaker/latest/dg/neo.html>`__ -
`Build Groundtruth
Datasets <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-getting-started.html>`__
- `Getting started with
SageMaker <https://medium.com/apache-mxnet/getting-started-with-sagemaker-ebe1277484c9>`__
