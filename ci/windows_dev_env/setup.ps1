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


$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
function Check-Call {
    param (
        [scriptblock]$ScriptBlock
    )
    Write-Host "Executing $ScriptBlock"
    & @ScriptBlock
    if (($lastexitcode -ne 0)) {
	Write-Error "Execution failed with $lastexitcode"
        exit $lastexitcode
    }
}
Check-Call { setx PATH "$($env:path);c:\Program Files\CMake\bin;" /m }
Set-ExecutionPolicy Bypass -Scope Process -Force
Invoke-WebRequest -Uri https://chocolatey.org/install.ps1 -OutFile install.ps1
./install.ps1
#Check-Call { C:\ProgramData\chocolatey\choco install python2 -y --no-progress }
Check-Call { C:\ProgramData\chocolatey\choco install python --version=3.7.0 --force -y --no-progress -r}
Check-Call { C:\Python37\python -m pip install --upgrade pip  }
Check-Call { C:\Python37\python -m pip install -r requirements.txt  }
#Check-Call { C:\Python27\python -m pip install --upgrade pip  }
#Check-Call { C:\Python27\python -m pip install -r requirements.txt  }

Check-Call { C:\ProgramData\chocolatey\choco install git -y -r --no-progress }
Check-Call { C:\ProgramData\chocolatey\choco install 7zip -y -r --no-progress }
Check-Call { C:\ProgramData\chocolatey\choco install cmake -y -r --no-progress }
Check-Call { C:\ProgramData\chocolatey\choco install ninja -y -r --no-progress }

# Deps
Check-Call { C:\Python37\python  windows_deps_headless_installer.py }

# Other software
#Check-Call { C:\ProgramData\chocolatey\choco install jom -y }
#Check-Call { C:\ProgramData\chocolatey\choco install mingw -y -r --no-progress }
Check-Call { C:\ProgramData\chocolatey\choco install javaruntime -y -r --no-progress }

# update path after all software is installed
refreshenv

Write-Output "End"
