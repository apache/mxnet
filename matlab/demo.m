%% Download sample image and model
if ~exist('cat.png', 'file')
  assert(~system('wget --no-check-certificate https://raw.githubusercontent.com/dmlc/mxnet.js/master/data/cat.png'));
end

if ~exist('model/Inception_BN-0039.params', 'file')
  assert(~system('wget --no-check-certificate https://s3.amazonaws.com/dmlc/model/inception-bn.tar.gz'));
  assert(~system('tar -zxvf inception-bn.tar.gz'))
end

%% Load the model
clear model
model = mxnet.model;
model.load('model/Inception_BN', 39);

%% Load and resize the image
img = imresize(imread('cat.png'), [224 224]);

%% Run prediction
model.forward(img);

%% Print all layers in the symbol
sym = model.parse_symbol();
layers = {};
for i = 1 : length(sym.nodes)
  if ~strcmp(sym.nodes{i}.op, 'null')
    layers{end+1} = sym.nodes{i}.name;
  end
end
layers

%% Extract feature

%%
