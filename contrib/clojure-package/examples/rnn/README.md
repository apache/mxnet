# rnn


Demonstration of LSTM RNN trainined using Obamas text

## Usage


run `./get_data.sh to download the training corpus as well as pretrained model.

Run `lein run` to start training the corpus from scratch for 2 epochs and then
show the result of training after 75 epochs (cpu)

You can control the devices you run on by doing:

`lein run :cpu 2` - This will run on 2 cpu devices
`lein run :gpu 1` - This will run on 1 gpu device
`lein run :gpu 2` - This will run on 2 gpu devices


