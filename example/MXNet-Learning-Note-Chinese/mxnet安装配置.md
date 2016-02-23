项目主页 https://github.com/dmlc/mxnet 

函数说明文档主页 http://mxnet.readthedocs.org/en/latest/doxygen/index.html 

帮助文档主页 http://mxnet.readthedocs.org/en/latest/

release版本主页 https://github.com/dmlc/mxnet/releases

Release版本直接根据redame操作即可，.sln文件可以直接运行（win7运行很可能会出现问题，如果出现进不了主程序这种情况，可以尝试升级系统到win10）

现在release版本有一些问题(2015.12.22)，某些函数在C++中调用可能会出现
```
error LNK2001: 无法解析的外部符号 Mxnet::Ndarray
```

需要在头文件中找到Ndarray的类，在类名之前加入MXNET_API，重新编译
另外本问题在已经有项目解决了，地址在https://github.com/hjk41/MxNet.cpp
，我还没有试过，大家可以去看一下。
####mxnet在vs2013中的配置

最简单的方法就是将编译生成好的.lib和.dll文件放在官方提供的release版本lib文件夹内，替换其中文件即可以
自己配置步骤如下

1. 包含目录
（mxnet，dmlc，mshadow，这三个放在同一个目录下，例如统一包含在include文件夹下，然后把包含这三个的目录加入到包含目录中），openblas（可以直接使用release版本提供的），opencv（建议使用3.0)，cuda，cudnn

2. 库目录，含有mxnet.lib文件的目录
3. 附加依赖项
 libmxnet.lib (必须加)
 kernel32.lib
 user32.lib
 gdi32.lib
 winspool.lib
 comdlg32.lib
 advapi32.lib
 shell32.lib
 ole32.lib
 oleaut32.lib
 uuid.lib
 odbc32.lib
 odbccp32.lib

4. 预处理器命令
 _SCL_SECURE_NO_WARNINGS (必须加)
 MSHADOW_IN_CXX11
 DMLC_USE_CXX11  (必须加，要求使用C++11)
 MSHADOW_USE_CBLAS (必须加，使用OpenBLAS加速)
 WIN32 (非必须，改为WIN64没有变化)
 MSHADOW_USE_CUDA=0 (必须加，不使用CUDA加速)
 _DEBUG
 _CONSOLE
 _LIB

5. 需要的加入到系统路径中的dll文件（这一步如果使用release版本就可以不做了，因为运行setupenv.cmd时已经完成了），有libmxnet.dll，openblas的dll文件，opencv的dll文件，cudnn的dll文件，cuda的一般在安装cuda时已经自动加入到了系统路径中

####自己从源文件编译
使用cmake

1. 选择vs2013，64位
2. 选择blas为openblas（release版本有提供，可以直接用），配置选项为，include文件夹正常，lib文件选择目录中的libopenblas.dll.a
3. 选择opencv文件夹，建议选择3.0版本，选择主文件夹下的build目录即可，cmake会根据这个目录直接配置的
4. cuda一般不用管


