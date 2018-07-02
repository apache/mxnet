# imclassification

This shows off how to do image classification with the module api

There is an example of the high level training api fit and also how to use multiple cpus/gpus

To see more examples of how to use different parts of the module api look at the module example

To run the example you must do

* `lein install` in the root of the main project directory
* cd into this project directory and do `lein run`. This will execute the cpu version.

You can control the devices you run on by doing:

`lein run :cpu 2` - This will run on 2 cpu devices
`lein run :gpu 1` - This will run on 1 gpu device
`lein run :gpu 2` - This will run on 2 gpu devices
