#!/bin/bash

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

set -e
set -x

pip install opencv-python
pip install scipy
pip install pygame

# Install arcade learning environment
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake ninja-build
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install sdl sdl_image sdl_mixer sdl_ttf portmidi
fi
git clone https://github.com/mgbellemare/Arcade-Learning-Environment || true
pushd .
cd Arcade-Learning-Environment
mkdir -p build
cd build
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON -GNinja ..
ninja
cd ..
pip install -e .
popd
cp Arcade-Learning-Environment/ale.cfg .

# Copy roms
git clone https://github.com/npow/atari || true
cp -R atari/roms .

