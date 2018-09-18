import mxnet as mx
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import loss
from mxnet.gluon import nn
import numpy as  np
import os
import sys
import gluonbook as gb


class Resnet1D(nn.Block):
    def __init__(self, num_channels, **kwargs):
        super(Resnet1D, self).__init__(**kwargs)
        self.conv1 = nn.Conv1D(num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1D(num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, x):
        y = nd.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return nd.relu(y + x)


class CBR(nn.Block):
    def __init__(self, num_channels, kernel_size=3, padding=1, **kwargs):
        super(CBR, self).__init__(**kwargs)
        self.conv = nn.Conv1D(num_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm()
        self.relu = nn.Activation('relu')

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SwapAxes(nn.Block):
    def __init__(self, dim1, dim2):
        super(SwapAxes, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return nd.swapaxes(x, self.dim1, self.dim2)

with mx.Context(mx.gpu(0)):
    model = nn.Sequential()
    model.add(SwapAxes(1,2),
              CBR(40, 1),
              CBR(40),
              CBR(40),
              nn.MaxPool1D(2),
              CBR(80, 1),
              CBR(80),
              CBR(80),
              nn.MaxPool1D(2),
              CBR(160, 1),
              CBR(160),
              CBR(160),
              CBR(160),
              nn.MaxPool1D(2),
              CBR(240, 1),
              # CBR(200),
              # CBR(200),
              # CBR(200),
              # nn.MaxPool1D(2),
              # CBR(300, 1)
              )
    for i in range(34):
        model.add(Resnet1D(240))

    model.add(# NCW
              nn.Conv1D(3000, 1, 1),
              # NWC
              SwapAxes(1, 2))


def ctc_loss(net, train_features, train_labels):
    preds = net(train_features)
    return loss.CTCLoss()(preds, train_labels)


def get_data_gen(data_dir, str2idx, batch_size=2):
    files = os.listdir(data_dir)
    new_files = []
    for f in files:
        if '.txt' in f:
            new_files.append(f)
    files = new_files
    files = list(set(list(map(lambda f:f.split('.')[0], files))))
    pooling_step = 8
    # np.random.seed(10)
    # while True:
    features = []
    labels = []
    input_len = []
    label_len = []
    np.random.shuffle(files)
    print('start one epoch')
    for idx in range(0, len(files)):
        try:
            feature = np.loadtxt(data_dir+'/'+files[idx]+'.txt') + 1
            #  mfcc.__call__(data_dir+'/'+files[new_idx]+'.wav')
            label = list(open(data_dir+'/'+files[idx]+'.wav.trn').readline().split('\n')[0].replace(' ', ''))
            label = np.array(list(map(lambda l:str2idx[l]+1, label)))
        except Exception as e:
            # print(e, files[idx])
            continue
        features.append(feature)
        labels.append(label)
        input_len.append(len(feature)/pooling_step-pooling_step)
        label_len.append(len(label))
        if len(features) == batch_size:
            maxLenFeature = max(list(map(len, features))) //pooling_step *pooling_step + pooling_step * 2
            maxLenLabel = max(list(map(len, labels)))
            featuresArr = np.zeros([len(features), maxLenFeature, 39], dtype=np.float32)
            labelsArr = np.ones([len(labels), maxLenLabel], dtype=np.float32) * 0  # (len(str2idx)+1)
            for idx in range(len(features)):
                featuresArr[idx, 0:len(features[idx]), :] = np.array(features[idx], dtype=np.float32)
                labelsArr[idx, :len(labels[idx])] = np.array(labels[idx], dtype=np.float32)
            yield featuresArr, labelsArr
            features = []
            labels = []
            input_len = []
            label_len = []


def get_str2idx(data_dir):
    files = os.listdir(data_dir)
    all_words = []
    str2idx = {}
    idx2str = {}
    for f in files:
        if 'trn' in f:
            all_words.extend(list(open(data_dir+'/'+f).readline().split('\n')[0].replace(' ', '')))
    all_words = list(set(all_words))
    for word, idx in enumerate(all_words):
        str2idx[word] = idx
        idx2str[str(idx)] = word
    return str2idx, idx2str


def get_iter(batch_size):
    data_dir = '/media/xmj/ubt_2t/中文语音识别/data_thchs30/data'
    train_iter = get_data_gen(data_dir, get_str2idx(data_dir)[1], batch_size)
    for x, y in train_iter:
        yield nd.array(x), nd.array(y)


class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0 # 当前的处理进度
    max_steps = 0 # 总共需要处理的次数
    max_arrow = 50 #进度条的长度
    infoDone = 'done'

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, loss, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%, loss:' + str(loss) + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0



my_ctcloss = loss.CTCLoss()
def train(net, num_epochs, lr, batch_size):
    with mx.Context(mx.gpu(0)):
        train_ls = []
        trainer = gluon.Trainer(net.collect_params(), 'adam',{
            'learning_rate': lr,
        })
        for epoch in range(num_epochs):
            max_steps = len(os.listdir('/media/xmj/ubt_2t/中文语音识别/data_thchs30/data'))//3 //batch_size
            process_bar = ShowProcess(max_steps, 'OK')
            train_iter = get_iter(batch_size)
            for x, y in train_iter:
                with autograd.record():
                    l = my_ctcloss(net(x), y) # .mean()
                l.backward()
                l = l.mean()
                trainer.step(batch_size)
                train_ls.append(l)
                process_bar.show_process(str(l)[2:8])
            if epoch % 1 == 0:
                # net.save_params('mxnetCnn'+str(epoch)+'.param')
                net.save_parameters('mxnetCnn'+str(epoch)+'.param')
                print('save to', epoch)
        return train_ls


def train_with_gb(net, num_epochs, lr, batch_size):
    with mx.Context(mx.gpu(0)):
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':lr})
        train_iter = get_iter(batch_size)
        test_iter = get_iter(batch_size//2)
        gb.train(train_iter, test_iter, net, my_ctcloss, trainer, mx.gpu(0), num_epochs )
ctx = [mx.gpu(0)] # , mx.cpu()]
model.initialize(init=init.Xavier(), ctx=ctx)
model.load_parameters('mxnetCnn20.param')
train(model, 10000, 0.01, 20)
# train_with_gb(model, 10000, 0.01, 2)