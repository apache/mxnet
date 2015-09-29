Build in Visual Studio 2013
=================

Prepare
---------

1. Download 3rdparty libraries form [BaiduYun](http://pan.baidu.com/s/1x7fw6) or [OneDrive](Not available now),  and extract the files into `3rdparty`

2. Download and install [Nov 2013 CTP](http://www.microsoft.com/en-us/download/details.aspx?id=41151), [PTVS](https://github.com/Microsoft/PTVS/releases).

3. Copy all files in `C:\Program Files (x86)\Microsoft Visual C++ Compiler Nov 2013 CTP` to `C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC` and overwrite all existed files (of course, backup them before copying).

Build
----------
1. Open mxnet.sln.

2. Switch the compile mode to Release and x64.

3. If you have MKL, please modify the path

4. Modify the cuda device compute capability defined in the settings (`mxnet properties` -> `CUDA C/C++` -> `Device` -> `Code Generation`) to your GPU's compute capability (such as compute_30,sm_30). You can look up for your GPU's compute capability in https://en.wikipedia.org/wiki/CUDA . Some general GPUs' compute capabilities are listed below.

5. Compile.

| GPU                                         | Compute Capability    |
| ------------------------------------------- |:---------------------:|
| GTX660, 680, 760, 770                       | compute_30,sm_30      |
| GTX780, Titan Z, Titan Black, K20, K40      | compute_35,sm_35      |
| GTX960, 980, Titan X                        | compute_52,sm_52      |