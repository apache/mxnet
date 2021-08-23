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

MXNet on the Cloud
==================

Deep learning can require extremely powerful hardware, often for
unpredictable durations of time. Moreover, *MXNet* can benefit from both multiple GPUs and multiple machines. Accordingly, cloud computing, as offered by AWS and others, is especially well suited to training deep learning models. Using AWS, we can rapidly fire up multiple machines with multiple GPUs each at will and maintain the resources for precisely the amount of time needed.

Here are some ways you can use MXNet on AWS:

1. Use [Amazon SageMaker](https://aws.amazon.com/sagemaker/developer-resources/)
1. Use the [AWS Deep Learning AMI with Conda](https://docs.aws.amazon.com/dlami/latest/devguide/overview-conda.html)
1. Use an [AWS Deep Learning Container](https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers.html)
1. Install MXNet on a [AWS Deep Learning Base AMI](https://docs.aws.amazon.com/dlami/latest/devguide/overview-base.html)
