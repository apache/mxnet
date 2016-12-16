@echo off
setlocal
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::: This script setup directories, dependencies for MXNET ::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


:::: Customizable variables ::::


:: conda environment name for MXNET, default to MXNET-vcversion
REM  set MXNET_CONDA_ENV=mxnet

:: which blas/lapack libraries will be used, default to openblas installed by conda
:: [1] mkl: download from https://software.intel.com/intel-mkl, install and set following two variables
REM  set INTEL_MKL_DIR=D:\\Intel\\SWTools\\compilers_and_libraries\\windows\\mkl\\
REM  set INTEL_COMPILER_DIR=D:\\Intel\\SWTools\\compilers_and_libraries\\windows\\compiler\\
:: [2] other: set path to the blas library and path to the laback library
:: both BLAS and LAPACK should be set even if they refer to the same library
:: take openblas for example: download latest release from https://github.com/xianyi/OpenBLAS/releases/latest
:: use mingw cross compiler tools in cygwin, since mingw windows native gfortrain is available in cygwin but not in msys2
:: compilation command in cygwin: make CC=x86_64-w64-mingw32-gcc FC=x86_64-w64-mingw32-gfortran CROSS_SUFFIX=x86_64-w64-mingw32-
:: please refer to openblas's README for detailed installation instructions
REM  set BLAS_LIBRARIES=D:\\Libraries\\lib\libopenblas.dll.a
REM  set LAPACK_LIBRARIES=D:\\Libraries\\lib\libopenblas.dll.a

:: where to find cudnn library
REM  set CUDNN_PATH=D:\NVIDIA\CUDNN\v5.1\bin\cudnn64_5.dll

:: whether update dependencies if already setup, default to not update
REM  set MXNET_UPDATE_DEPS=

::::  End of customization  ::::


set ECHO_PREFIX=+++++++

set MXNET_SETUP_HAS_CUDA=0
set MXNET_SETUP_HAS_CUDNN=0
:::: validate msvc version  ::::

if "%VisualStudioVersion%" == "" (
  if not "%VS140COMNTOOLS%" == "" ( call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" x64 && goto :VS_SETUP)
  if not "%VS120COMNTOOLS%" == "" ( call "%VS120COMNTOOLS%..\..\VC\vcvarsall.bat" x64 && goto :VS_SETUP)
  if not "%VS110COMNTOOLS%" == "" ( call "%VS110COMNTOOLS%..\..\VC\vcvarsall.bat" x64 && goto :VS_SETUP)
  if not "%VS100COMNTOOLS%" == "" ( call "%VS100COMNTOOLS%..\..\VC\vcvarsall.bat" x64 && goto :VS_SETUP)
  if not "%VS90COMNTOOLS%"  == "" ( call "%VS90COMNTOOLS%..\..\VC\vcvarsall.bat"  x64 && goto :VS_SETUP)
)
:VS_SETUP

if "%VisualStudioVersion%" == "" (
  echo %ECHO_PREFIX% Can not find environment variable VisualStudioVersion, msvc is not setup porperly
  goto :FAIL
)

set MXNET_VS_VERSION=%VisualStudioVersion:.0=%

if "%PreferredToolArchitecture%" == "x64" (
  if "%CommandPromptType%" == "Cross" (
    if "%Platform%" == "ARM" set MXNET_VS_PLATFORM=amd64_arm
    if "%Platform%" == "X86" set MXNET_VS_PLATFORM=amd64_x86
  )
) else (
  if "%CommandPromptType%" == "Cross" (
    if "%Platform%" == "ARM" set MXNET_VS_PLATFORM=x86_arm
    if "%Platform%" == "x64" set MXNET_VS_PLATFORM=x86_amd64
  )
  if "%CommandPromptType%" == "Native" (
    if "%Platform%" == "X64" set MXNET_VS_PLATFORM=x64
  )
  if "%Platform%"   == ""    set MXNET_VS_PLATFORM=x86
)

if     "%MXNET_VS_PLATFORM%" == "x86"                       set MXNET_VS_TARGET=x86
if not "%MXNET_VS_PLATFORM%" == "%MXNET_VS_PLATFORM:_x86=%" set MXNET_VS_TARGET=x86
if not "%MXNET_VS_PLATFORM%" == "%MXNET_VS_PLATFORM:_arm=%" set MXNET_VS_TARGET=arm
if     "%MXNET_VS_TARGET%"   == ""                          set MXNET_VS_TARGET=x64

::::    Setup directories   ::::

set MXNET_DISTRO=%~dp0.
set MXNET_DISTRO=%MXNET_DISTRO%\..
if "%MXNET_INSTALL_DIR%" == "" set MXNET_INSTALL_DIR=%MXNET_DISTRO%\build
set MXNET_INSTALL_BIN=%MXNET_INSTALL_DIR%\bin
set MXNET_INSTALL_LIB=%MXNET_INSTALL_DIR%\lib
set MXNET_INSTALL_INC=%MXNET_INSTALL_DIR%\include
set MXNET_INSTALL_ROC=%MXNET_INSTALL_DIR%\luarocks
set MXNET_INSTALL_TOOLS=%MXNET_INSTALL_DIR%\tools
if not exist %MXNET_INSTALL_BIN% md %MXNET_INSTALL_BIN%
if not exist %MXNET_INSTALL_LIB% md %MXNET_INSTALL_LIB%
if not exist %MXNET_INSTALL_INC% md %MXNET_INSTALL_INC%

echo %ECHO_PREFIX% MXNET will be installed under %MXNET_INSTALL_DIR% with %MXNET_LUA_SOURCE%, vs%MXNET_VS_VERSION% %MXNET_VS_PLATFORM%
echo %ECHO_PREFIX% Bin: %MXNET_INSTALL_BIN%
echo %ECHO_PREFIX% Lib: %MXNET_INSTALL_LIB%
echo %ECHO_PREFIX% Inc: %MXNET_INSTALL_INC%

::::   Setup dependencies   ::::

:: has blas/lapack?
if not "%INTEL_MKL_DIR%" == "" if exist %INTEL_MKL_DIR% set MXNET_BLAS=MKL
if not "%BLAS_LIBRARIES%" == "" if exist %BLAS_LIBRARIES% set MXNET_BLAS=Open

:: has cuda?
for /f "delims=" %%i in ('where nvcc') do (
  set NVCC_CMD=%%i
  goto :AFTER_NVCC
)
:AFTER_NVCC
if not "%NVCC_CMD%" == "" set MXNET_SETUP_HAS_CUDA=1

:: has conda?
for /f "delims=" %%i in ('where conda') do (
  set CONDA_CMD=%%i
  goto :AFTER_CONDA
)
:AFTER_CONDA

if "%CONDA_CMD%" == "" (
  echo %ECHO_PREFIX% Can not find conda, some dependencies can not be resolved
  if not "%MXNET_SETUP_HAS_BLAS%" == "1" (
    echo %ECHO_PREFIX% Can not install MXNET, please either specify the blas library or install conda
    goto :FAIL
  )
  goto :NO_CONDA
)

set MXNET_CONDA_INFO=%TEMP%\check_conda_info_for_MXNET.txt
conda info > %MXNET_CONDA_INFO%
if "%MXNET_VS_TARGET%" == "x64" set MXNET_CONDA_PLATFORM=win-64
if "%MXNET_VS_TARGET%" == "arm" set MXNET_CONDA_PLATFORM=win-64
if "%MXNET_VS_TARGET%" == "x86" set MXNET_CONDA_PLATFORM=win-32

findstr "%MXNET_CONDA_PLATFORM%" "%MXNET_CONDA_INFO%" >nul
if errorlevel 1 (
  echo %ECHO_PREFIX% %MXNET_VS_TARGET% MXNET requires %MXNET_CONDA_PLATFORM% conda, installation will continue without conda
  goto :NO_CONDA
)

if %MXNET_VS_VERSION% GEQ 14 ( set CONDA_VS_VERSION=14&& goto :CONDA_SETUP )
if %MXNET_VS_VERSION% GEQ 10 ( set CONDA_VS_VERSION=10&& goto :CONDA_SETUP )
set CONDA_VS_VERSION=9

:CONDA_SETUP

if "%MXNET_CONDA_ENV%" == "" set MXNET_CONDA_ENV=mxnet-vc%CONDA_VS_VERSION%

echo %ECHO_PREFIX% Createing conda environment '%MXNET_CONDA_ENV%' for MXNET dependencies
conda create -n %MXNET_CONDA_ENV% -c conda-forge vc=%CONDA_VS_VERSION% --yes

set CONDA_DIR=%CONDA_CMD:\Scripts\conda.exe=%
set MXNET_CONDA_LIBRARY=%CONDA_DIR%\envs\%MXNET_CONDA_ENV%\Library
set MXNET_CONDA_LIBRARY=%MXNET_CONDA_LIBRARY:\=\\%
set PATH=%MXNET_CONDA_LIBRARY%\bin;%PATH%;
set NEW_PATH=%CONDA_DIR%\Scripts;%MXNET_CONDA_LIBRARY%\bin;%NEW_PATH%

set MXNET_CONDA_PKGS=%TEMP%\check_conda_packages_for_MXNET.txt
conda list -n %MXNET_CONDA_ENV% > %MXNET_CONDA_PKGS%

:: has cmake?
for /f "delims=" %%i in ('where cmake') do (
  set CMAKE_CMD=%%i
  goto :AFTER_CMAKE
)
if "%CMAKE_CMD%" == "" (
  echo %ECHO_PREFIX% Installing cmake by conda
  conda install -n %MXNET_CONDA_ENV% -c conda-forge cmake --yes
)
:AFTER_CMAKE

:: need openblas?
if not "%MXNET_BLAS%" == "Open" goto :CONDA_INSTALL_OPENBLAS
goto :AFTER_OPENBLAS

:CONDA_INSTALL_OPENBLAS
findstr "openblas" "%MXNET_CONDA_PKGS%" >nul
if errorlevel 1 (
  echo %ECHO_PREFIX% Installing openblas by conda, since there is no blas library specified
  if "%MXNET_VS_TARGET%" == "x64" conda install -n %MXNET_CONDA_ENV% -c mutirri openblas --yes || goto :Fail
)
set MXNET_BLAS=Open

:AFTER_OPENBLAS
if "%MXNET_VS_TARGET%" == "x64" (
  if "%OpenBLAS_INCLUDE_DIR%" == "" set OpenBLAS_INCLUDE_DIR=%MXNET_CONDA_LIBRARY%\\include\\
  if "%OpenBLAS_LIB%" == "" set OpenBLAS_LIB=%MXNET_CONDA_LIBRARY%\\lib\\libopenblas.dll.a
)

:: other dependencies
findstr "opencv" "%MXNET_CONDA_PKGS%" >nul
if errorlevel 1 set MXNET_DEPENDENCIES=%MXNET_DEPENDENCIES% opencv
findstr "numpy" "%MXNET_CONDA_PKGS%" >nul
if errorlevel 1 set MXNET_DEPENDENCIES=%MXNET_DEPENDENCIES% numpy
REM findstr "7za" "%MXNET_CONDA_PKGS%" >nul
REM if errorlevel 1 set MXNET_DEPENDENCIES=%MXNET_DEPENDENCIES% 7za
findstr "ninja" "%MXNET_CONDA_PKGS%" >nul
if errorlevel 1 set MXNET_DEPENDENCIES=%MXNET_DEPENDENCIES% ninja

if not "%MXNET_DEPENDENCIES%" == "" (
  echo %ECHO_PREFIX% Installing %MXNET_DEPENDENCIES% by conda for MXNET
  conda install -n %MXNET_CONDA_ENV% -c conda-forge %MXNET_DEPENDENCIES% --yes
)

:: make symbolic link to workaround opencv cmakefile bug
echo %ECHO_PREFIX% Install symblic link for opencv
mklink /D %CONDA_DIR%\envs\x64\vc14\lib %MXNET_CONDA_LIBRARY%\lib
mklink /D %CONDA_DIR%\envs\x64\vc14\bin %MXNET_CONDA_LIBRARY%\bin

:NO_CONDA
if exist "%MXNET_CONDA_INFO%" del /q %MXNET_CONDA_INFO%
if exist "%MXNET_CONDA_PKGS%" del /q %MXNET_CONDA_PKGS%

SET PATH=%PATH%;%MXNET_CONDA_LIBRARY%\bin
set NEW_PATH=%NEW_PATH%;%MXNET_CONDA_LIBRARY%\bin

::::   download graphviz   ::::

REM echo %ECHO_PREFIX% Downloading graphviz for graph package
REM cd %MXNET_DISTRO%\build
REM wget -nc https://github.com/mahkoCosmo/GraphViz_x64/raw/master/graphviz-2.38_x64.tar.gz --no-check-certificate -O graphviz-2.38_x64.tar.gz
REM 7z x graphviz-2.38_x64.tar.gz -y && 7z x graphviz-2.38_x64.tar -ographviz-2.38_x64 -y >NUL
REM if not exist %MXNET_INSTALL_BIN%\graphviz md %MXNET_INSTALL_BIN%\graphviz
REM copy /y %MXNET_DISTRO%\build\graphviz-2.38_x64\bin\ %MXNET_INSTALL_BIN%\graphviz\

REM set NEW_PATH=%NEW_PATH%;%MXNET_INSTALL_BIN%\graphviz

echo %ECHO_PREFIX% Build libmxnet.dll
cd %MXNET_DISTRO%

cmake -Wno-dev -DUSE_CUDNN=%MXNET_SETUP_HAS_CUDNN% -DUSE_CUDA=%MXNET_SETUP_HAS_CUDA% -DCMAKE_PREFIX_PATH=%MXNET_CONDA_LIBRARY% -DBLAS="%MXNET_BLAS%" -DOpenBLAS_HOME=%MXNET_CONDA_LIBRARY% -DOpenBLAS_INCLUDE_DIR=%OpenBLAS_INCLUDE_DIR% -DOpenBLAS_LIB=%OpenBLAS_LIB% -DOpenCV_LIB_PATH=%MXNET_CONDA_LIBRARY%\\lib\\ -DOpenCV_INCLUDE_DIRS=%MXNET_CONDA_LIBRARY%\\include\\ -DOpenCV_CONFIG_PATH=%MXNET_CONDA_LIBRARY% -G "Ninja" -DCMAKE_BUILD_TYPE=Release -H. -Bbuild
if errorlevel 1 goto :FAIL
cmake --build build --config Release
if errorlevel 1 goto :FAIL

echo %ECHO_PREFIX% Install libmxnet.dll
cd %MXNET_DISTRO%\python
python setup.py install
if errorlevel 1 goto :FAIL

echo %ECHO_PREFIX% Setup succeed!
echo %ECHO_PREFIX% Add the following path to your system path
echo %NEW_PATH%
goto :END

:FAIL
echo %ECHO_PREFIX% Setup fail!

:END
