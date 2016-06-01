import mxnet as mx
import numpy as np
import minibatch


class ROIIter(mx.io.DataIter):
    def __init__(self, roidb, ctx, batch_size=2, shuffle=False, mode='train', work_load_list=None):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :return: ROIIter
        """
        super(ROIIter, self).__init__()

        self.roidb = roidb
        self.ctx = ctx
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.work_load_list = work_load_list
        if self.mode != 'train':
            assert self.batch_size == 1

        self.cur = 0
        self.size = len(roidb)
        self.index = np.arange(self.size)
        self.num_classes = self.roidb[0]['gt_overlaps'].shape[1]

        self.batch = None
        self.data = None
        self.label = None
        self.get_batch()

    @property
    def provide_data(self):
        return [('data', self.data[0].shape), ('rois', self.data[1].shape)]

    @property
    def provide_label(self):
        return [('cls_prob_label', self.label[0].shape),
                ('bbox_loss_target', self.label[1].shape),
                ('bbox_loss_inside_weight', self.label[2].shape),
                ('bbox_loss_outside_weight', self.label[3].shape)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            if self.mode == 'train':
                return mx.io.DataBatch(data=self.data, label=self.label,
                                       pad=self.getpad(), index=self.getindex(),
                                       provide_data=self.provide_data, provide_label=self.provide_label)
            else:
                return mx.io.DataBatch(data=self.data, label=self.label,
                                       pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        if self.mode == 'train':
            self.batch = self._get_train_batch()
            self.data = [mx.nd.array(self.batch['data']), mx.nd.array(self.batch['rois'])]
            self.label = [mx.nd.array(self.batch['labels']), mx.nd.array(self.batch['bbox_targets']),
                mx.nd.array(self.batch['bbox_inside_weights']), mx.nd.array(self.batch['bbox_outside_weights'])]
        else:
            self.batch = self._get_test_batch()
            self.data = {'data': self.batch['data'],
                         'rois': self.batch['rois']}
            self.label = {}

    def _get_train_batch(self):
        """
        utilize minibatch sampling, e.g. 2 images and 64 rois per image
        :return: training batch (e.g. 128 samples)
        """
        cur_from = self.cur
        cur_to = cur_from + self.batch_size
        if cur_to <= self.size:
            roidb = [self.roidb[i] for i in range(cur_from, cur_to)]
        else:
            pad = cur_to - self.size
            roidb = self.roidb[cur_from:] + self.roidb[:pad]
        batch = minibatch.get_minibatch(roidb, self.num_classes, self.ctx, self.work_load_list)
        return batch

    def _get_test_batch(self):
        """
        testing batch is composed of 1 image, all rois
        :return: testing batch
        """
        roidb = self.roidb[self.index[self.cur]]
        roidb = [roidb]
        batch = minibatch.get_testbatch(roidb, self.num_classes)
        return batch
