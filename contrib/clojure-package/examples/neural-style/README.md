<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# neural-style

An example of neural style transfer

## Usage

use the `download.sh` script to get the params file and the input and output file

Then use `lein run`

The output images will be stored in the output directory. Please feel free to play with the params at the top of the file


This example only works on 1 device (cpu) right now

If you are running on AWS you will need to setup X11 for graphics
`sudo apt install xauth x11-apps`

then relogin in `ssh -X -i creds ubuntu@yourinstance`


_Note: This example is not working all the way - it needs some debugging help_


