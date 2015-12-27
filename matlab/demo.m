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
img = single(img) - 120;
%% Run prediction
pred = model.forward(img);

%% load the labels
labels = {};
fid = fopen('model/synset.txt', 'r');
assert(fid >= 0);
tline = fgetl(fid);
while ischar(tline)
  labels{end+1} = tline;
  tline = fgetl(fid);
end
fclose(fid);

%% find the predict label
[p, i] = max(pred);
fprintf('the best result is %s, with probability %f\n', labels{i}, p)

%% Print the last 10 layers in the symbol

sym = model.parse_symbol();
layers = {};
for i = 1 : length(sym.nodes)
  if ~strcmp(sym.nodes{i}.op, 'null')
    layers{end+1} = sym.nodes{i}.name;
  end
end
fprintf('layer name: %s\n', layers{end-10:end})

%% Extract feature from internal layers

feas = model.forward(img, {'max_pool_5b_pool', 'global_pool', 'fc'});
feas(:)

%% If GPU is available
% feas = model.forward(img, 'gpu', 0, {'max_pool_5b_pool', 'global_pool', 'fc'});
% feas(:)
