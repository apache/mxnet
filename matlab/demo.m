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

%% Assumes model symbol and parameters already downloaded using .sh script

%% Load the model
clear model
format compact
model = mxnet.model;
model.load('data/Inception-BN', 126);

%% Load and resize the image
img = imresize(imread('data/cat.png'), [224 224]);
img = single(img) - 120;
%% Run prediction
pred = model.forward(img);

%% load the labels
labels = {};
fid = fopen('data/synset.txt', 'r');
assert(fid >= 0);
tline = fgetl(fid);
while ischar(tline)
  labels{end+1} = tline;
  tline = fgetl(fid);
end
fclose(fid);

%% Print top 5 predictions
fprintf('Top 5 predictions: \n');
[p, i] = sort(pred, 'descend');
for x = 1:5
    fprintf('    %2.2f%% - %s\n', p(x)*100, labels{i(x)} );
end

%% Print the last 10 layers in the symbol
fprintf('\nLast 10 layers in the symbol: \n');
sym = model.parse_symbol();
layers = {};
for i = 1 : length(sym.nodes)
  if ~strcmp(sym.nodes{i}.op, 'null')
    layers{end+1} = sym.nodes{i}.name;
  end
end
fprintf('    layer name: %s\n', layers{end-10:end})


%% Extract feature from internal layers
fprintf('\nExtract feature from internal layers using CPU forwarding: \n');
feas = model.forward(img, {'max_pool_5b_pool', 'global_pool', 'fc1'});
feas(:)

%% If GPU is available
fprintf('\nExtract feature from internal layers using GPU forwarding: \n');
feas = model.forward(img, 'gpu', 0, {'max_pool_5b_pool', 'global_pool', 'fc1'});
feas(:)
