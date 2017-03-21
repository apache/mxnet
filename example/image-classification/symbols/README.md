# Symbol

This fold contains definition of various networks. To add a new network, please
use the following format.

## Python

- A file implements one network proposed in a paper, with the network name as the
filename.
- Mention the paper and the modifications made if any at the beginning
of the file.
- Indicate how to reproduce the accuracy numbers in the paper if it is not straightforward
- Provide a function `get_symbol()` that return the network
