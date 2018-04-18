Time major data layout for RNN
==============================

This example demonstrates an RNN implementation with Time-major layout. This implementation shows 1.5x-2x speedups compared to Batch-major RNN.
	
As example of Batch-major RNN is available in MXNet [RNN Bucketing example](https://github.com/apache/incubator-mxnet/tree/master/example/rnn/bucketing)
	
## Running the example
- Prerequisite: an instance with GPU compute resources is required to run MXNet RNN
- Make the shell script ```get_ptb_data.sh``` executable:
    ```bash 
    chmod +x get_ptb_data.sh
    ```
- Run ```get_ptb_data.sh``` to download the PTB dataset, and follow the instructions to review the license:
    ```bash
    ./get_ptb_data.sh
    ```
    The PTB data sets will be downloaded into ./data directory, and available for the example to train on.
- Run the example:
    ```bash
    python python rnn_cell_demo.py
    ```
    
    If everything goes well, console will plot training speed and perplexity that you can compare to the batch major RNN.
