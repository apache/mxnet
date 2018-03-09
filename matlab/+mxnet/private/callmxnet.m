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

function callmxnet(func, varargin)
%CALLMXNET call mxnet functions

if ~libisloaded('libmxnet')
  cur_pwd = pwd;
  mxnet_root = [fileparts(mfilename('fullpath')), '/../../../'];
  cd(mxnet_root);
  mxnet_root = pwd;
  cd(cur_pwd);
  
  assert(exist([mxnet_root, '/lib/libmxnet.so'   ], 'file') == 2 || ...
         exist([mxnet_root, '/lib/libmxnet.dylib'], 'file') == 2 || ...
         exist([mxnet_root, '/lib/libmxnet.dll'  ], 'file') == 2, ...
         'you need to build mxnet first');
  assert(exist([mxnet_root, '/include/mxnet/c_predict_api.h']) == 2, ...
         'failed to find c_predict_api.h')
  addpath([mxnet_root, '/lib'])
  addpath([mxnet_root, '/include/mxnet'])

  [err, warn] = loadlibrary('libmxnet', 'c_predict_api.h');
  assert(isempty(err));
  if warn, warn, end
end

assert(ischar(func))
ret = calllib('libmxnet', func, varargin{:});
assert(ret == 0)
end
