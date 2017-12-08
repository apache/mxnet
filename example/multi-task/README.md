# Mulit-task learning example
 
This is a simple example to show how to use mxnet for multi-task learning. It uses MNIST as an example and mocks up the multi-label task.

## Usage
First, you need to write a multi-task iterator on your own. The iterator needs to generate multiple labels according to your applications, and the label names should be specified in the `provide_label` function, which needs to be consist with the names of output layers. 

Then, if you want to show metrics of different tasks separately, you need to write your own metric class and specify the `num` parameter. In the `update` function of metric, calculate the metrics separately for different tasks.

The example script uses gpu as device by default, if gpu is not available for your environment, you can change `device` to be `mx.cpu()`.
