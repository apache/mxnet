Build and Installation
======================
- You can clone the mxnet from the [github repo](https://github.com/dmlc/mxnet)
- After you clone the repo, update the submodules by
```bash
git submodule init
git submodule update
```
- Copy [make/config.mk](../make/config.mk) to the project root, modify according to your desired setting.
- Type ```make``` in the root folder.


Install Python Package
----------------------
After you build the mxnet, you can install python package by
```bash
cd python
python setup.py install
```
