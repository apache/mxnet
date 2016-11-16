Memory Cost of Deep Nets under Different Allocations
====================================================
This folder contains a script to show the memory cost of different allocation strategies,
discussed in [Note on Memory Optimization](http://mxnet.io/architecture/note_memory.html).

We use inception-bn as an example, with batch size of 32.

How to See the cost
-------------------
The possible options are gathered together in the [Makefile](Makefile).
Type the following command to see the allocation cost. Look for the
```Final message Total x MB allocated```
- ```make no_optimization```
  - Shows the cost without any optimization.
- ```make with_inplace```
  - Shows the cost with inplace optimization.
- ```make with_sharing```
  - Shows the cost with memory allocating algorithm for sharing.
- ```make with_both```
  - Shows the cost of memory allocation with both inplace and sharing optimization.
- ```make forward_only```
  - Shows the cost of when we only want to run forward pass.

Notes
-----
- You can change the symbol in the [inception_memcost.py](inception_memcost.py) to the net you interested in.
- You will need to install mxnet or type make on the root folder before use the script.
- The estimation is only on space cost of intermediate node.
  - The cost of temporal workspace is not estimated, so you will likely need more memory when running real nets.
- The estimation does real allocation on CPU, the plan is the same on GPU.
