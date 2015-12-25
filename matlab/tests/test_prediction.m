%% prepare

addpath('..')

if ~exist('mnist-test.mat', 'file')
  system('wget --no-check-certificate https://github.com/dmlc/web-data/raw/master/mxnet/matlab/mnist-test.mat');
end

if ~exist('model/mnist-lenet-0-0010.params', 'file')
  system('wget --no-check-certificate https://github.com/dmlc/web-data/raw/master/mxnet/matlab/mnist-lenet.tar.gz');
  system('tar -zxf mnist-lenet.tar.gz');
end

%% load data and model

load mnist-test
clear model
model = mxnet.model;
model.load('model/mnist-lenet-0', 10);

%% predict

err = 0;
batch = 1000;
for i = 1 : length(Y) / batch
  ix = (i-1)*batch+1 : i*batch;
  x = X(:,:,:,ix);
  pred = model.forward(x, 'gpu', 0);
  [~, k] = max(pred);
  err = err + nnz(k ~= Y(ix)');
end

err = err / length(Y);
fprintf('prediction error: %f\n', err)

%%
% ix = 1:2;
% x = X(:,:,:,ix);
% pred = model.forward(x, {'pooling1', 'fullyconnected1', 'softmax'});

%%
% batch = 1000;
% e = 0;
% for i = 1 : batch
%   x = single(X(:,:,:,i));
%   pred = model.forward(x);
%   [~, k] = max(pred);
%   e = e + (k == Y(i));
% end

% e / batch

% %% load data
% load cifar10-test.mat
% img_mean = [123.68, 116.779, 103.939];

% %%
% clear model
% model = mxnet.model;
% model.load('model/cifar10-incept-bn-0', 20);

% %%
% batch = 100;
% x = zeros(28,28,3,batch);
% for i = 1 : batch
%   x(:,:,:,i) = single(imresize(X(:,:,:,i), [28, 28]));
%   x = x(:,:,[3 2 1],:);
% end
% % x = permute(x, [2 1 3 4]);

% x = x - 120;
% % for i = 1 : 3
% %   x(:,:,i,:) = x(:,:,i,:) - img_mean(i);
% % end


% pred = model.forward(x, 'gpu', 0);

% [~,i] = max(reshape(pred(:), 10, batch));
% nnz(i' == Y(1:batch)) / length(i)

% %%

% batch = 100;
% e = 0;
% for i = 1 : batch
%   x = single(imresize(X(:,:,:,i), [28, 28])) - 120;
%   for j = 1 : 3
%     x(:,:,j) = x(:,:,j);
%   end
%   pred = model.forward(x);
%   [~, k] = max(pred);
%   e = e + (k == Y(i));
% end

% e / batch


% %% load bin

% a = fopen('mean.bin', 'r');
% yy = fread(a, 14, '*int32');
% mm = fread(a, inf, '*single');
% fclose(a)
% % nn = mm(14:end-6);
