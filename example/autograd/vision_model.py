from __future__ import division

from mxnet import initializer as init
from mxnet.foo import nn

# Helpers
def conv3x3(filters, stride, in_filters):
    return nn.Conv2D(filters, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_filters=in_filters)


# Blocks
class BasicBlockV1(nn.HybridLayer):
    def __init__(self, filters, stride, downsample=False, in_filters=0, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3(filters, stride, in_filters)
            self.bn1 = nn.BatchNorm(num_features=in_filters)
            self.conv2 = conv3x3(filters, 1, filters)
            self.bn2 = nn.BatchNorm(num_features=filters)
            if downsample:
                self.conv_ds = nn.Conv2D(filters, kernel_size=1, strides=stride,
                                         use_bias=False, in_filters=in_filters)
                self.bn_ds = nn.BatchNorm(num_features=filters)
            self.downsample = downsample

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.Activation(x, act_type='relu')

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.conv_ds(x)
            residual = self.bn_ds(residual)

        out = residual + out
        out = F.Activation(out, act_type='relu')

        return out


class BottleneckV1(nn.HybridLayer):
    def __init__(self, filters, stride, downsample=False, in_filters=0, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(filters=filters//4, kernel_size=1, strides=1,
                                   in_filters=in_filters)
            self.bn1 = nn.BatchNorm(num_features=filters//4)
            self.conv2 = conv3x3(filters//4, stride, filters//4)
            self.bn2 = nn.BatchNorm(num_features=filters//4)
            self.conv3 = nn.Conv2D(filters=filters, kernel_size=1, strides=1, in_filters=filters//4)
            self.bn3 = nn.BatchNorm(num_features=filters)
            if downsample:
                self.conv_ds = nn.Conv2D(filters, kernel_size=1, strides=stride,
                                         use_bias=False, in_filters=in_filters)
                self.bn_ds = nn.BatchNorm(num_features=filters)
            self.downsample = downsample

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.Activation(out, act_type='relu')

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.Activation(out, act_type='relu')

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.conv_ds(x)
            residual = self.bn_ds(residual)

        out = out + residual

        out = F.Activation(out, act_type='relu')
        return out


class BasicBlockV2(nn.HybridLayer):
    def __init__(self, filters, stride, downsample=False, in_filters=0, **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        with self.name_scope():
            self.bn1 = nn.BatchNorm(num_features=in_filters)
            self.conv1 = conv3x3(filters, stride, in_filters)
            self.bn2 = nn.BatchNorm(num_features=filters)
            self.conv2 = conv3x3(filters, 1, filters)
            if downsample:
                self.downsample = nn.Conv2D(filters, 1, stride, use_bias=False,
                                            in_filters=in_filters)
            else:
                self.downsample = None

    def hybrid_forward(self, F, x):
        if not self.downsample:
            residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        return x + residual


class BottleneckV2(nn.HybridLayer):
    def __init__(self, filters, stride, downsample=False, in_filters=0, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        with self.name_scope():
            self.bn1 = nn.BatchNorm(num_features=in_filters)
            self.conv1 = conv3x3(filters//4, 1, in_filters)
            self.bn2 = nn.BatchNorm(num_features=filters//4)
            self.conv2 = conv3x3(filters//4, stride, filters//4)
            self.bn3 = nn.BatchNorm(num_features=filters//4)
            self.conv3 = conv3x3(filters, 1, filters//4)
            if downsample:
                self.downsample = nn.Conv2D(filters, 1, stride, use_bias=False,
                                            in_filters=in_filters)
            else:
                self.downsample = None

    def hybrid_forward(self, F, x):
        if not self.downsample:
            residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv3(x)

        return x + residual


# Nets
class ResnetV1(nn.HybridLayer):
    def __init__(self, block, classes, layers, filters, thumbnail, **kwargs):
        super(ResnetV1, self).__init__(**kwargs)
        with self.name_scope():
            assert len(layers) == len(filters) - 1
            self._thumbnail = thumbnail
            if thumbnail:
                self.conv0 = conv3x3(filters[0], 1, 3)
            else:
                self.conv0 = nn.Conv2D(filters[0], 7, 2, 3, use_bias=False,
                                       in_filters=3)
                self.bn0 = nn.BatchNorm(num_features=filters[0])
                self.pool0 = nn.MaxPool2D(3, 2, 1)

            self.body = nn.HSequential()
            with self.body.name_scope():
                for i, num_layer in enumerate(layers):
                    stride = 1 if i == 0 else 2
                    self.body.add(self._make_layer(block, num_layer, filters[i+1],
                                                   stride, in_filters=filters[i]))

            self.pool1 = nn.GlobalAvgPool2D()
            self.dense1 = nn.Dense(classes, in_units=filters[-1])

    def _make_layer(self, block, layers, filters, stride, in_filters=0):
        layer = nn.HSequential()
        with layer.name_scope():
            layer.add(block(filters, stride, True, in_filters=in_filters))
            for _ in range(layers-1):
                layer.add(block(filters, 1, False, in_filters=filters))
        return layer

    def hybrid_forward(self, F, x):
        x = self.conv0(x)
        if not self._thumbnail:
            x = self.bn0(x)
            x = F.Activation(x, act_type='relu')
            x = self.pool0(x)

        x = self.body(x)

        x = self.pool1(x)
        x = x.reshape((0, -1))
        x = self.dense1(x)

        return x


class ResnetV2(nn.HybridLayer):
    def __init__(self, block, classes, layers, filters, thumbnail, **kwargs):
        super(ResnetV2, self).__init__(**kwargs)
        with self.name_scope():
            assert len(layers) == len(filters) - 1
            self._thumbnail = thumbnail
            self.bn_data = nn.BatchNorm(num_features=3, scale=False, center=False)
            if thumbnail:
                self.conv0 = conv3x3(filters[0], 1, 3)
            else:
                self.conv0 = nn.Conv2D(filters[0], 7, 2, 3, use_bias=False,
                                       in_filters=3)
                self.bn0 = nn.BatchNorm(num_features=filters[0])
                self.pool0 = nn.MaxPool2D(3, 2, 1)

            self.body = nn.HSequential()
            with self.body.name_scope():
                in_filters = filters[0]
                for i, num_layer in enumerate(layers):
                    stride = 1 if i == 0 else 2
                    self.body.add(self._make_layer(block, num_layer, filters[i+1],
                                                   stride, in_filters=in_filters))
                    in_filters = filters[i+1]

            self.bn1 = nn.BatchNorm(num_features=in_filters)
            self.pool1 = nn.GlobalAvgPool2D()
            self.dense1 = nn.Dense(classes, in_units=in_filters)

    def _make_layer(self, block, layers, filters, stride, in_filters=0):
        layer = nn.HSequential()
        with layer.name_scope():
            layer.add(block(filters, stride, True, in_filters=in_filters))
            for _ in range(layers-1):
                layer.add(block(filters, 1, False, in_filters=filters))
        return layer

    def hybrid_forward(self, F, x):
        x = self.bn_data(x)
        x = self.conv0(x)
        if not self._thumbnail:
            x = self.bn0(x)
            x = F.Activation(x, act_type='relu')
            x = self.pool0(x)

        x = self.body(x)

        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        x = self.pool1(x)
        x = x.reshape((0, -1))
        x = self.dense1(x)

        return x


class VGG(nn.HybridLayer):
    def __init__(self, classes, layers, filters, batch_norm, **kwargs):
        super(VGG, self).__init__(**kwargs)
        with self.name_scope():
            assert len(layers) == len(filters)
            self.features = self._make_features(layers, filters, batch_norm)
            self.classifier = nn.HSequential()
            with self.classifier.name_scope():
                self.classifier.add(nn.Dense(4096, activation='relu',
                                             kernel_initializer=init.Normal(),
                                             bias_initializer='zeros'))
                self.classifier.add(nn.Dropout(rate=0.5))
                self.classifier.add(nn.Dense(4096, activation='relu',
                                             kernel_initializer=init.Normal(),
                                             bias_initializer='zeros'))
                self.classifier.add(nn.Dropout(rate=0.5))
                self.classifier.add(nn.Dense(classes,
                                             kernel_initializer=init.Normal(),
                                             bias_initializer='zeros'))

    def _make_features(self, layers, filters, batch_norm):
        featurizer = nn.HSequential()
        with featurizer.name_scope():
            for i, num in enumerate(layers):
                for _ in range(num):
                    featurizer.add(nn.Conv2D(filters[i], kernel_size=3, padding=1,
                                             kernel_initializer=init.Xavier(rnd_type='gaussian',
                                                                            factor_type='out',
                                                                            magnitude=2),
                                             bias_initializer='zeros'))
                    if batch_norm:
                        featurizer.add(nn.BatchNorm())
                    featurizer.add(nn.Activation('relu'))
                featurizer.add(nn.MaxPool2D(strides=2))
        return featurizer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2], [16, 16, 32, 64]),
               34: ('basic_block', [3, 4, 6, 3], [16, 16, 32, 64]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
            13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
            16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
            19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}

resnet_net_versions = [ResnetV1, ResnetV2]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                         {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]


# Constructor
def get_resnet(version, num_layers, classes, use_thumbnail):
    block_type, layers, filters = resnet_spec[num_layers]
    resnet = resnet_net_versions[version]
    block = resnet_block_versions[version][block_type]
    return resnet(block, classes, layers, filters, use_thumbnail)

def get_vgg(num_layers, classes, batch_norm):
    layers, filters = vgg_spec[num_layers]
    return VGG(classes, layers, filters, batch_norm)


# Convenience constructors
def resnet18_v1(classes, use_thumbnail=False):
    return get_resnet(0, 18, classes, use_thumbnail)
def resnet34_v1(classes, use_thumbnail=False):
    return get_resnet(0, 34, classes, use_thumbnail)
def resnet50_v1(classes, use_thumbnail=False):
    return get_resnet(0, 50, classes, use_thumbnail)
def resnet101_v1(classes, use_thumbnail=False):
    return get_resnet(0, 101, classes, use_thumbnail)
def resnet152_v1(classes, use_thumbnail=False):
    return get_resnet(0, 152, classes, use_thumbnail)

def resnet18_v2(classes, use_thumbnail=False):
    return get_resnet(1, 18, classes, use_thumbnail)
def resnet34_v2(classes, use_thumbnail=False):
    return get_resnet(1, 34, classes, use_thumbnail)
def resnet50_v2(classes, use_thumbnail=False):
    return get_resnet(1, 50, classes, use_thumbnail)
def resnet101_v2(classes, use_thumbnail=False):
    return get_resnet(1, 101, classes, use_thumbnail)
def resnet152_v2(classes, use_thumbnail=False):
    return get_resnet(1, 152, classes, use_thumbnail)

def vgg11(classes, batch_norm=False):
    return get_vgg(11, classes, batch_norm)
def vgg13(classes, batch_norm=False):
    return get_vgg(13, classes, batch_norm)
def vgg16(classes, batch_norm=False):
    return get_vgg(16, classes, batch_norm)
def vgg19(classes, batch_norm=False):
    return get_vgg(19, classes, batch_norm)

def get_vision_model(name, **kwargs):
    models = {'resnet18_v1': resnet18_v1,
              'resnet34_v1': resnet34_v1,
              'resnet50_v1': resnet50_v1,
              'resnet101_v1': resnet101_v1,
              'resnet152_v1': resnet152_v1,
              'resnet18_v2': resnet18_v2,
              'resnet34_v2': resnet34_v2,
              'resnet50_v2': resnet50_v2,
              'resnet101_v2': resnet101_v2,
              'resnet152_v2': resnet152_v2,
              'vgg11': vgg11,
              'vgg13': vgg13,
              'vgg16': vgg16,
              'vgg19': vgg19}
    assert name in models, 'Model %s is not supported'%name
    return models[name](**kwargs)
