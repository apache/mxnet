@ECHO off
SETLOCAL

::===============================================================================
:: Copyright 2019-2020 Intel Corporation
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
::===============================================================================

:process_arguments
IF "%1" == "/TESTKIND" SET "TESTKIND=%2"
IF "%1" == "/BUILDDIR" SET "BUILDDIR=%2"
IF "%1" == "/MODE" SET "MODE=%2"
IF "%1" == "/REPORTDIR" SET "REPORTDIR=%2"

SHIFT
SHIFT
IF NOT "%1" == "" GOTO process_arguments

SET "CTEST_OPTS=--output-on-failure"

IF "%TESTKIND%" == "gtest" SET "CTEST_OPTS=%CTEST_OPTS% -E benchdnn"
IF "%TESTKIND%" == "benchdnn" SET "CTEST_OPTS=%CTEST_OPTS% -R benchdnn --verbose" 
IF NOT "%TESTKIND%" == "benchdnn" IF NOT "%TESTKIND%" == "gtest" (
    ECHO "Error: unknown test kind: %TESTKIND%"
    EXIT 1
)

IF NOT "%REPORTDIR%" == "" SET "GTEST_OUTPUT=%REPORTDIR%\report\test_report.xml"

SET "PATH=%BUILDDIR%\src\%MODE%;%PATH%"
SET "LIB=%BUILDDIR%\src\%MODE%;%LIB%"

ECHO "CTEST OPTIONS: %CTEST_OPTS%"

CD /D %BUILDDIR%

ctest %CTEST_OPTS%

echo "DONE"
