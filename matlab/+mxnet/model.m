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

% TODO
end

function outputs = forward(obj, imgs)
%FORWARD perform forward
%
% OUT = MODEL.FORWARD(imgs) returns the forward (prediction) outputs of a list
% of images, where imgs can be either a single image with the format
%
%   width x height x channel
%
% which is return format of `imread` or a list of images with format
%
%   width x height x channel x num_images
%
% MODEL.FORWARD(imgs, 'gpu', [0, 1]) uses GPU 0 and 1 for prediction
%
% MODEL.FORWARD(imgs, {'conv4', 'conv5'}) extract outputs for two internal layers
%
% Examples
%
%   % load and resize an image
%   img = imread('test.jpg')
%   img = imresize(img, [224 224])
%   % get the softmax output
%   out = model.forward(img)
%   % get the output of two internal layers
%   out = model.forward(img, {'conv4', 'conv5'})
%   % use gpu 0
%   out = model.forward(img, 'gpu', 0)
%   % use two gpus for a image list
%   imgs(:,:,:,1) = img1
%   imgs(:,:,:,2) = img2
%   out = model.forward(imgs, 'gpu', [0,1])


% TODO
r = 0;
end

end
end
