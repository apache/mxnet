## An Example of Subgraph Operator and Graph Partitioning

In this example, we demonstrated how to trigger graph partitioning and group adjacent whitelisted operators
into subgraph operators. The type of the subgraph operators for storing subgraphs are determined by an command
line option: `--subgraph-backend`. Currently, only the default subgraph operator is supported, which means
the subgraphs consisting of whitelisted operators are executed by `CachedOp`, an internal graph executor,
while the whole graph is executed by `GraphExecutor`.

In the context of integrating a specific accelerator backend with MXNet, the operator whitelist is defined
by that accelerator and will be used for partitioning computational graphs.
Since here we are using the `default` subgraph backend which does not have a
pre-defined operator whitelist, we defined it in Python as the following:
```python
op_names = ['BatchNorm', 'Convolution', 'Pooling', 'Activation']
```
This will effectively group adjacent operators in the above list into `_default_subgraph_op` nodes
in computational graphs.

Here is a command that you can try to run this example:
```bash
python imagenet_inference.py --model imagenet1k-resnet-152 --dataset ./data --subgraph-backend default --ctx gpu --num-inference-batches 10
```