classdef model < handle
%MODEL MXNet model, supports load and forward

properties
% The symbol definition, in json format
  symbol
% parameter weights
  params
% whether or not print info
  verbose
end

properties (Access = private)
% mxnet predictor
  predictor
% the previous input size
  prev_input_size
% the previous device id
  prev_dev_id
% the previous device type (cpu or gpu)
  prev_dev_type
end

methods
  function obj = model()
  %CONSTRUCTOR
  obj.predictor = libpointer('voidPtr', 0);
  obj.prev_input_size = zeros(1,4);
  obj.verbose = 1;
  obj.prev_dev_id = -1;
  obj.prev_dev_type = -1;
  end

  function delete(obj)
  %DESTRUCTOR
  obj.free_predictor();
  end

  function load(obj, model_prefix, num_epoch)
  %LOAD load model from files
  %
  % A mxnet model is stored into two files. The first one contains the symbol
  % definition in json format. While the second one stores all weights in binary
  % format. For example, if we save a model using the prefix 'model/vgg19' at
  % epoch 8, then we will get two files. 'model/vgg19-symbol.json' and
  % 'model/vgg19-0009.params'
  %
  % model_prefix : the string model prefix
  % num_epoch : the epoch to load
  %
  % Example:
  %   model = mxnet.model
  %   model.load('outptu/vgg19', 8)

  % read symbol
  obj.symbol = fileread([model_prefix, '-symbol.json']);

  % read params
  fid = fopen(sprintf('%s-%04d.params', model_prefix, num_epoch), 'rb');
  assert(fid ~= 0);
  obj.params = fread(fid, inf, '*ubit8');
  fclose(fid);
  end

  function json = parse_symbol(obj)
  json = parse_json(obj.symbol);
  end


  function outputs = forward(obj, input, varargin)
  %FORWARD perform forward
  %
  % OUT = MODEL.FORWARD(input) returns the forward (prediction) outputs of a list
  % of input examples
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

  % check arguments
  dev_type = 1; % cpu in default
  dev_id = 0;
  while length(varargin) > 0
    if ischar(varargin{1}) && strcmp(varargin{1}, 'gpu')
      assert(length(varargin) > 1, 'arg error: no gpu id')
      assert(isnumeric(varargin{2}))
      dev_type = 2;
      dev_id = varargin{2};
      varargin = varargin(3:end);
    end
  end

  siz = size(input);
  assert(length(siz) >= 2);

  % convert from matlab order (col-major) into c order (row major):
  input = obj.convert_ndarray(input);


  if length(siz) ~= length(obj.prev_input_size) || ...
        any(siz ~= obj.prev_input_size) || ...
        dev_type ~= obj.prev_dev_type || ...
        length(dev_id) ~= length(obj.prev_dev_id) || ...
        any(dev_id ~= obj.prev_dev_id)
    obj.free_predictor()
  end
  obj.prev_input_size = siz;
  obj.prev_dev_type = dev_type;
  obj.prev_dev_id = dev_id;

  if obj.predictor.Value == 0
    fprintf('create predictor with input size ');
    fprintf('%d ', siz);
    fprintf('\n');
    csize = [ones(1, 4-length(siz)), siz(end:-1:1)];
    callmxnet('MXPredCreatePartialOut', obj.symbol, ...
              libpointer('voidPtr', obj.params), ...
              length(obj.params), ...
              int32(dev_type), int32(dev_id), ...
              1, {'data'}, ...
              uint32([0, 4]), ...
              uint32(csize), ...
              0, {}, ...
              obj.predictor);
  end

  % feed input
  callmxnet('MXPredSetInput', obj.predictor, 'data', single(input(:)), uint32(numel(input)));
  % forward
  callmxnet('MXPredForward', obj.predictor);

  % get output size
  out_dim = libpointer('uint32Ptr', 0);
  out_shape = libpointer('uint32PtrPtr', ones(4,1));
  callmxnet('MXPredGetOutputShape', obj.predictor, 0, out_shape, out_dim);
  assert(out_dim.Value <= 4);
  out_siz = out_shape.Value(1:out_dim.Value);
  out_siz = double(out_siz(end:-1:1))';

  % get output
  out = libpointer('singlePtr', single(zeros(out_siz)));

  callmxnet('MXPredGetOutput', obj.predictor, 0, ...
            out, uint32(prod(out_siz)));

  % TODO convert from c order to matlab order...
  outputs = out.Value;
  % outputs = obj.convert_ndarray(out.Value);
  if length(out_siz) > 2
    outputs = convert_ndarray(outputs);
  end
  end
end

methods (Access = private)
  function free_predictor(obj)
  % free the predictor
  if obj.predictor.Value ~= 0
    callmxnet('MXPredFree', obj.predictor);
    obj.predictor = libpointer('voidPtr', 0);
  end
  end

  function Y = convert_ndarray(obj, X)
  % convert between matlab's col major and c's row major
  siz = size(X);
  Y = permute(X, [2 1 3:length(siz)]);
  end
end

end
