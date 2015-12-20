classdef model
%MODEL MXNet model, supports load and forward

properties
% The symbol definition, in json format
  symbol
% parameter weights
  param
end

methods

function load(obj, model_prefix, num_epoch)
%LOAD load model from files
%
% A mxnet model is stored into two files. The first one contains the symbol
% definition in json format. While the second one stores all weights in binary
% format. For example, if we save a model using the prefix 'output/vgg19' at
% epoch 8, then we will get two files. 'output/vgg19-symbol.json' and
% 'output/vgg19-0009.params'
%
% model_prefix : the string model prefix
% num_epoch : the epoch to load
%
% Example:
%   model = mxnet.model
%   model.load('outptu/vgg19', 8)

end

function r = forward(obj, r)
%FORWARD perform forward
r = 0;
end

end
end
