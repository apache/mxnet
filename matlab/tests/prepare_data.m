% Licensed to the Apache Software Foundation (ASF) under one
% or more contributor license agreements.  See the NOTICE file
% distributed with this work for additional information
% regarding copyright ownership.  The ASF licenses this file
% to you under the Apache License, Version 2.0 (the
% "License"); you may not use this file except in compliance
% with the License.  You may obtain a copy of the License at
%
%   http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing,
% software distributed under the License is distributed on an
% "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
% KIND, either express or implied.  See the License for the
% specific language governing permissions and limitations
% under the License.

%% download cifar10 dataset
system('wget https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz')
system('tar -xzvf cifar-10-matlab.tar.gz')
load cifar-10-batches-mat/test_batch.mat

%% convert test dataset of cifar10, and save
X = reshape(data', [32, 32, 3, 10000]);
X = permute(X, [2 1 3 4]);
Y = labels + 1;


save cifar10-test X Y
%% preview one picture
imshow(imresize(X(:,:,:,2), [128, 128]))

%%

!wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
!wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
!gunzip t10k-images-idx3-ubyte.gz
!gunzip t10k-labels-idx1-ubyte.gz

%%

fid = fopen('t10k-images-idx3-ubyte', 'r');
d = fread(fid, inf, '*uint8');
fclose(fid);
X = reshape(d(17:end), [28 28 1 10000]);
X = permute(X, [2 1 3 4]);

fid = fopen('t10k-labels-idx1-ubyte', 'r');
d = fread(fid, inf, '*uint8');
fclose(fid);
Y = d(9:end) + 1;

save mnist-test X Y
