# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

7z x -y windows_package.7z

# set default output encoding to utf8
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

$env:MXNET_HOME = [System.IO.Path]::GetFullPath('.\windows_package')
$env:JULIA_URL = "https://julialang-s3.julialang.org/bin/winnt/x64/0.7/julia-0.7.0-win64.exe"
$env:JULIA_DEPOT_PATH = [System.IO.Path]::GetFullPath('.\julia-depot')

$JULIA_DIR = [System.IO.Path]::GetFullPath('.\julia07')
$JULIA = "$JULIA_DIR\bin\julia"

# Download most recent Julia Windows binary
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
(New-Object System.Net.WebClient).DownloadFile($env:JULIA_URL, "julia-binary.exe")
if (! $?) { Throw ("Error on downloading Julia Windows binary") }

# Run installer silently, output to C:\julia07\julia
Start-Process -Wait "julia-binary.exe" -ArgumentList "/S /D=$JULIA_DIR"
if (! $?) { Throw ("Error on installing Julia") }

& $JULIA -e "using InteractiveUtils; versioninfo()"

dir

$src='
    using Pkg
    Pkg.activate(".\\julia")
    Pkg.build()
    Pkg.test()
'

$src > .\ci-build.jl

# Redirect all stderr output to stdout,
# since Julia loggers output stuffs to stderr.
# Then, stderr triggers powershell NativeCommandError.
& $JULIA .\ci-build.jl 2>&1 | %{ "$_" }
if ($LastExitCode -eq 1) { Throw ("Error") }
