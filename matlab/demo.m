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
