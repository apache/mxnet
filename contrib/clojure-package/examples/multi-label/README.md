# multi-label

This is a quick example of doing multi-label classification.
It involves using a proxy to implement the DataIter to make a custom
data iterator for MNIST

To run
`lein run`. This will execute the cpu version.

You can control the devices you run on by doing:

`lein run :cpu` - This will run on 1 cpu device
`lein run :gpu` - This will run on 1 gpu device

This example only works on 1 device



