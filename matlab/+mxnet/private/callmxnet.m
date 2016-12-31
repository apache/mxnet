function callmxnet(func, varargin)
%CALLMXNET call mxnet functions

if ~libisloaded('libmxnet')
  cur_pwd = pwd;
  mxnet_root = [fileparts(mfilename('fullpath')), '/../../../'];
  cd(mxnet_root);
  mxnet_root = pwd;
  cd(cur_pwd);
  assert(exist([mxnet_root, '/lib/libmxnet.so'], 'file') == 2 || ...
         exist([mxnet_root, '/lib/libmxnet.dylib'], 'file') == 2 || ...
         exist([mxnet_root, '/lib/libmxnet.dll'], 'file') == 2, ...
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
