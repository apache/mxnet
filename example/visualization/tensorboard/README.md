Visualizing Training MNIST Model
=============================

This folder contains an example of logging MXNet data for visualization in TensorBoard
in the process of training the MNIST model using Gluon interfaces. To run the example,
type `python mnist.py` in the terminal. While the training program is running, launch
TensorBoard by typing the following command under the current path:
```bash
tensorboard --logdir=./logs --host=127.0.0.1 --port=8888
```
Then open the browser and enter the address `127.0.0.1:8888`.
You would be able to see the figures of training/validation accuracy curves,
histograms of the gradients of all the parameters evolving with time, and training images
of the first mini-batch of each epoch.
