rem Licensed to the Apache Software Foundation (ASF) under one
rem or more contributor license agreements.  See the NOTICE file
rem distributed with this work for additional information
rem regarding copyright ownership.  The ASF licenses this file
rem to you under the Apache License, Version 2.0 (the
rem "License"); you may not use this file except in compliance
rem with the License.  You may obtain a copy of the License at
rem
rem   http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing,
rem software distributed under the License is distributed on an
rem "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
rem KIND, either express or implied.  See the License for the
rem specific language governing permissions and limitations
rem under the License.

@echo off
setlocal
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::: This script setup directories, dependencies for MXNET ::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


:::: Customizable variables ::::


:: conda environment name for MXNET, default to MXNET-vcversion
REM  set MXNET_CONDA_ENV=mxnet

:: which to find MKL, default to openblas installed by conda
:: mkl: download from https://software.intel.com/intel-mkl, install and set following two variables
REM  set INTEL_MKL_DIR=D:\\Intel\\SWTools\\compilers_and_libraries\\windows\\mkl\\

:: where to find cudnn library
REM  set CUDNN_ROOT=D:\NVIDIA\CUDNN\v5.1\

::::  End of customization  ::::


set ECHO_PREFIX=+++++++

set MXNET_SETUP_HAS_CUDA=0
set MXNET_SETUP_HAS_CUDNN=0
:::: validate msvc version  ::::

if "%VisualStudioVersion%" == "" (
  if not "%VS140COMNTOOLS%" == "" ( call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" x64 && goto :VS_SETUP)
REM  Not Supported yet due to dependencies
REM  if not "%VS120COMNTOOLS%" == "" ( call "%VS120COMNTOOLS%..\..\VC\vcvarsall.bat" x64 && goto :VS_SETUP)
REM  if not "%VS110COMNTOOLS%" == "" ( call "%VS110COMNTOOLS%..\..\VC\vcvarsall.bat" x64 && goto :VS_SETUP)
REM  if not "%VS100COMNTOOLS%" == "" ( call "%VS100COMNTOOLS%..\..\VC\vcvarsall.bat" x64 && goto :VS_SETUP)
REM  if not "%VS90COMNTOOLS%"  == "" ( call "%VS90COMNTOOLS%..\..\VC\vcvarsall.bat"  x64 && goto :VS_SETUP)
)
:VS_SETUP

if "%VisualStudioVersion%" == "" (
  echo %ECHO_PREFIX% Can not find environment variable VisualStudioVersion, msvc is not setup porperly
  echo %ECHO_PREFIX% Please Install Visual Studio from:
  echo %ECHO_PREFIX% https://go.microsoft.com/fwlink/?LinkId=691978&clcid=0x409
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

echo %ECHO_PREFIX% MXNET will be installed under %MXNET_INSTALL_DIR%, vs%MXNET_VS_VERSION% %MXNET_VS_PLATFORM%

::::   Setup dependencies   ::::

:: has blas/lapack?
if exist %INTEL_MKL_DIR% set MXNET_BLAS=MKL
if "%MXNET_BLAS%" == "" set MXNET_BLAS=Open

:: has cuda?
for /f "delims=" %%i in ('where nvcc') do (
  set NVCC_CMD=%%i
  goto :AFTER_NVCC
)
:AFTER_NVCC
if not "%NVCC_CMD%" == "" set MXNET_SETUP_HAS_CUDA=1

:: has cudnn
if exist %CUDNN_ROOT% set MXNET_SETUP_HAS_CUDNN=1

:: has conda?
for /f "delims=" %%i in ('where conda') do (
  set CONDA_CMD=%%i
  goto :AFTER_CONDA
)
:AFTER_CONDA

if "%CONDA_CMD%" == "" (
  echo %ECHO_PREFIX% Can not find conda, some dependencies can not be resolved
  echo %ECHO_PREFIX% Please install Miniconda with Python 3.5 from here:
  echo %ECHO_PREFIX% http://conda.pydata.org/miniconda.html

  goto :FAIL
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
set PATH=%MXNET_CONDA_LIBRARY:\\=\%\bin;%PATH%;
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

:: has patch?
for /f "delims=" %%i in ('where patch') do (
  set PATCH_CMD=%%i
  goto :AFTER_PATCH
)
if "%PATCH_CMD%" == "" (
  echo %ECHO_PREFIX% Installing patch by conda
  conda install -n %MXNET_CONDA_ENV% patch --yes
)
:AFTER_PATCH

:: need openblas?
if "%MXNET_BLAS%" == "Open" goto :CONDA_INSTALL_OPENBLAS
goto :AFTER_OPENBLAS

:CONDA_INSTALL_OPENBLAS
findstr "openblas" "%MXNET_CONDA_PKGS%" >nul
if errorlevel 1 (
  echo %ECHO_PREFIX% Installing openblas by conda
  if "%MXNET_VS_TARGET%" == "x64" conda install -n %MXNET_CONDA_ENV% -c ukoethe openblas --yes || goto :Fail
  CALL :PATCH_OPENBLAS
)

if "%MXNET_VS_TARGET%" == "x64" (
  if "%OpenBLAS_INCLUDE_DIR%" == "" set OpenBLAS_INCLUDE_DIR=%MXNET_CONDA_LIBRARY%\\include\\
  if "%OpenBLAS_LIB%" == "" set OpenBLAS_LIB=%MXNET_CONDA_LIBRARY%\\lib\\libopenblas.lib
)
:AFTER_OPENBLAS

if "%MXNET_BLAS%" == "MKL" goto:CONDA_INSTALL_MKL
goto :AFTER_MKL

:CONDA_INSTALL_MKL
if "%MKL_INCLUDE_DIR%" == "" set MKL_INCLUDE_DIR=%INTEL_MKL_DIR%\\include\\
if "%MKL_LIB%" == "" set MKL_LIB=%INTEL_MKL_DIR%\\lib\\
:AFTER_MKL

:: other dependencies
findstr "opencv" "%MXNET_CONDA_PKGS%" >nul
if errorlevel 1 set MXNET_DEPENDENCIES=%MXNET_DEPENDENCIES% opencv
findstr "numpy" "%MXNET_CONDA_PKGS%" >nul
if errorlevel 1 set MXNET_DEPENDENCIES=%MXNET_DEPENDENCIES% numpy
findstr "cython" "%MXNET_CONDA_PKGS%" >nul
if errorlevel 1 set MXNET_DEPENDENCIES=%MXNET_DEPENDENCIES% cython
findstr "numpy" "%MXNET_CONDA_PKGS%" >nul
if errorlevel 1 set MXNET_DEPENDENCIES=%MXNET_DEPENDENCIES% numpy

if not "%MXNET_DEPENDENCIES%" == "" (
  echo %ECHO_PREFIX% Installing %MXNET_DEPENDENCIES% by conda for MXNET
  conda install -n %MXNET_CONDA_ENV% -c conda-forge %MXNET_DEPENDENCIES% --yes
)

:: make symbolic link to workaround opencv cmakefile bug
echo %ECHO_PREFIX% Install symblic link for opencv
mkdir %CONDA_DIR%\envs\x64\vc14\
mklink /D %CONDA_DIR%\envs\x64\vc14\lib %MXNET_CONDA_LIBRARY:\\=\%\lib
mklink /D %CONDA_DIR%\envs\x64\vc14\bin %MXNET_CONDA_LIBRARY:\\=\%\bin

:NO_CONDA
if exist "%MXNET_CONDA_INFO%" del /q %MXNET_CONDA_INFO%
if exist "%MXNET_CONDA_PKGS%" del /q %MXNET_CONDA_PKGS%

::::   download graphviz   ::::

REM echo %ECHO_PREFIX% Downloading graphviz for graph package
REM cd %MXNET_DISTRO%\build
REM wget -nc https://github.com/mahkoCosmo/GraphViz_x64/raw/master/graphviz-2.38_x64.tar.gz --no-check-certificate -O graphviz-2.38_x64.tar.gz
REM 7z x graphviz-2.38_x64.tar.gz -y && 7z x graphviz-2.38_x64.tar -ographviz-2.38_x64 -y >NUL
REM if not exist %MXNET_INSTALL_BIN%\graphviz md %MXNET_INSTALL_BIN%\graphviz
REM copy /y %MXNET_DISTRO%\build\graphviz-2.38_x64\bin\ %MXNET_INSTALL_BIN%\graphviz\

REM set NEW_PATH=%NEW_PATH%;%MXNET_INSTALL_BIN%\graphviz

echo %ECHO_PREFIX% Build libmxnet.dll
cd /D %MXNET_DISTRO%

SET CMAKE_OPT=%CMAKE_OPT% -DUSE_CUDA=%MXNET_SETUP_HAS_CUDA%
SET CMAKE_OPT=%CMAKE_OPT% -DUSE_CUDNN=%MXNET_SETUP_HAS_CUDNN%
if %MXNET_SETUP_HAS_CUDNN%=="1" SET CMAKE_OPT=%CMAKE_OPT% -DCUDNN_ROOT="%CUDNN_ROOT%"
SET CMAKE_OPT=%CMAKE_OPT% -DBLAS="%MXNET_BLAS%"
if "%MXNET_BLAS%"=="Open" SET CMAKE_OPT=%CMAKE_OPT% -DOpenBLAS_HOME="%MXNET_CONDA_LIBRARY%" -DOpenBLAS_INCLUDE_DIR="%OpenBLAS_INCLUDE_DIR%" -DOpenBLAS_LIB="%OpenBLAS_LIB%"
if "%MXNET_BLAS%"=="MKL" SET CMAKE_OPT=%CMAKE_OPT% -DMKL_HOME="%INTEL_MKL_DIR%" -DMKL_INCLUDE_DIR="%MKL_INCLUDE_DIR%" -DMKL_LIB="%MKL_LIB%"

SET CMAKE_OPT=%CMAKE_OPT% -DOpenCV_LIB_PATH="%MXNET_CONDA_LIBRARY%\\lib\\" -DOpenCV_INCLUDE_DIRS="%MXNET_CONDA_LIBRARY%\\include\\" -DOpenCV_CONFIG_PATH="%MXNET_CONDA_LIBRARY%"

cmake -Wno-dev %CMAKE_OPT% -DCMAKE_PREFIX_PATH="%MXNET_CONDA_LIBRARY%" -G "Visual Studio 14 2015 Win64" -DUSE_PROFILER=1 -DCMAKE_BUILD_TYPE=Release -H. -Bbuild
if errorlevel 1 goto :FAIL
msbuild build\mxnet.sln /t:Build /p:Configuration=Release;Platform=x64 /m
if errorlevel 1 goto :FAIL

echo %ECHO_PREFIX% Install libmxnet.dll
cd /D %MXNET_DISTRO%\python
python setup.py install
if errorlevel 1 goto :FAIL

set NEW_PATH=%NEW_PATH:\\=\%
echo %ECHO_PREFIX% Setup succeed!
echo %ECHO_PREFIX% Add the following path to your system path or Run env.cmd per cmd window
echo %NEW_PATH%
echo @SET PATH=%%PATH%%;%NEW_PATH%>%MXNET_DISTRO%\env.cmd
goto :END

:PATCH_OPENBLAS
echo %ECHO_PREFIX% Apply patch to fix openblas 2.15

echo --- openblas_config.h   2016-12-20 11:09:11.722445000 +0800>%TEMP%\openblas.diff
echo +++ openblas_config.h   2016-12-20 11:01:21.347244600 +0800>>%TEMP%\openblas.diff
echo @@ -109,7 +109,7 @@>>%TEMP%\openblas.diff
echo     structure as fallback (see Clause 6.2.5.13 of the C99 standard). */>>%TEMP%\openblas.diff
echo  #if (defined(__STDC_IEC_559_COMPLEX__) ^|^| __STDC_VERSION__ ^>= 199901L ^|^| \>>%TEMP%\openblas.diff
echo       (__GNUC__ ^>= 3 ^&^& !defined(__cplusplus)) ^|^| \>>%TEMP%\openblas.diff
echo -     _MSC_VER ^>= 1800) // Visual Studio 2013 supports complex>>%TEMP%\openblas.diff
echo +     (_MSC_VER ^>= 1800 ^&^& !defined(__cplusplus))) // Visual Studio 2013 supports complex>>%TEMP%\openblas.diff
echo    #define OPENBLAS_COMPLEX_C99>>%TEMP%\openblas.diff
echo  #ifndef __cplusplus>>%TEMP%\openblas.diff
echo    #include ^<complex.h^>>>%TEMP%\openblas.diff
cd /D %MXNET_CONDA_LIBRARY:\\=\%\include
patch -p0 -i %TEMP%\openblas.diff
DEL %TEMP%\openblas.diff
GOTO :eof

:FAIL
echo %ECHO_PREFIX% Setup fail!

:END

