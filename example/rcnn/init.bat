cd /d %~dp0
mkdir .\data
cd rcnn\cython
python setup_windows.py build_ext --inplace
python setup_windows_cuda.py build_ext --inplace
pause
cd ..\pycocotools
python setup_windows.py build_ext --inplace
cd ..\..
pause
