## Instructions

This shows off how to use the module api.

There are examples of:
 - high level api of training and prediction
 - intermediate level api with save and loading from checkpoints
 - examples of how to iteratate through the batch and calculate accuracy and predict manually.

To run the example you must do

* `lein install` in the root of the main project directory
* cd into this project directory and do `lein run`. This will execute the cpu version.

You can control the devices you run on by doing:

`lein run :cpu 2` - This will run on 2 cpu devices
`lein run :gpu 1` - This will run on 1 gpu device
`lein run :gpu 2` - This will run on 2 gpu devices


