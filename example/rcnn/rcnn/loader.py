import mxnet as mx
import numpy as np
import minibatch
from config import config
from mxnet.executor_manager import _split_input_slice
from helper.processing.image_processing import tensor_vstack


class ROIIter(mx.io.DataIter):
    def __init__(self, roidb, batch_size=2, shuffle=False, mode='train', ctx=None, work_load_list=None):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param mode: control returned info
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :return: ROIIter
        """
        super(ROIIter, self).__init__()

        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list

        self.cur = 0
        self.size = len(roidb)
        self.index = np.arange(self.size)
        self.num_classes = self.roidb[0]['gt_overlaps'].shape[1]
        self.reset()

        self.batch = None
        self.data = None
        self.label = None
        self.get_batch()
        self.data_name = ['data', 'rois']
        self.label_name = ['label', 'bbox_target', 'bbox_inside_weight', 'bbox_outside_weight']

    @property
    def provide_data(self):
        if self.mode == 'train':
            return [('data', self.data[0].shape), ('rois', self.data[1].shape)]
        else:
            return [(k, v.shape) for k, v in self.data.items()]

    @property
    def provide_label(self):
        if self.mode == 'train':
            return [('label', self.label[0].shape),
                    ('bbox_target', self.label[1].shape),
                    ('bbox_inside_weight', self.label[2].shape),
                    ('bbox_outside_weight', self.label[3].shape)]
        else:
            return [(k, v.shape) for k, v in self.data.items()]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if config.TRAIN.ASPECT_GROUPING:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                inds = np.reshape(inds, (-1, 2))
                row_perm = np.random.permutation(np.arange(inds.shape[0]))
                inds = np.reshape(inds[row_perm, :], (-1, ))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
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
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        if self.mode == 'test':
            self.data, self.label = minibatch.get_minibatch(roidb, self.num_classes, self.mode)
        else:
            work_load_list = self.work_load_list
            ctx = self.ctx
            if work_load_list is None:
                work_load_list = [1] * len(ctx)
            assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
                "Invalid settings for work load. "
            slices = _split_input_slice(self.batch_size, work_load_list)

            data_list = []
            label_list = []
            for islice in slices:
                iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
                data, label = minibatch.get_minibatch(iroidb, self.num_classes, self.mode)
                data_list.append(data)
                label_list.append(label)

            all_data = dict()
            for key in data_list[0].keys():
                all_data[key] = tensor_vstack([batch[key] for batch in data_list])

            all_label = dict()
            for key in label_list[0].keys():
                all_label[key] = tensor_vstack([batch[key] for batch in label_list])

            self.data = [mx.nd.array(all_data['data']),
                         mx.nd.array(all_data['rois'])]
            self.label = [mx.nd.array(all_label['label']),
                          mx.nd.array(all_label['bbox_target']),
                          mx.nd.array(all_label['bbox_inside_weight']),
                          mx.nd.array(all_label['bbox_outside_weight'])]


class AnchorLoader(mx.io.DataIter):
    def __init__(self, feat_sym, roidb, batch_size=1, shuffle=False, mode='train', ctx=None, work_load_list=None,
                 feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), allowed_border=0):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param mode: control returned info
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :return: AnchorLoader
        """
        super(AnchorLoader, self).__init__()

        self.feat_sym = feat_sym
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border

        self.cur = 0
        self.size = len(roidb)
        self.index = np.arange(self.size)
        self.num_classes = self.roidb[0]['gt_overlaps'].shape[1]
        self.reset()

        self.batch = None
        self.data = None
        self.label = None
        self.get_batch()
        self.data_name = ['data', 'im_info']
        self.label_name = ['label', 'bbox_target', 'bbox_inside_weight', 'bbox_outside_weight']

    @property
    def provide_data(self):
        if self.mode == 'train':
            return [('data', self.data[0].shape)]
        else:
            return [(k, v.shape) for k, v in self.data.items()]

    @property
    def provide_label(self):
        if self.mode == 'train':
            return [('label', self.label[0].shape),
                    ('bbox_target', self.label[1].shape),
                    ('bbox_inside_weight', self.label[2].shape),
                    ('bbox_outside_weight', self.label[3].shape)]
        else:
            return [(k, v.shape) for k, v in self.data.items()]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if config.TRAIN.ASPECT_GROUPING:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                inds = np.reshape(inds, (-1, 2))
                row_perm = np.random.permutation(np.arange(inds.shape[0]))
                inds = np.reshape(inds[row_perm, :], (-1, ))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
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
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        if self.mode == 'test':
            self.data, self.label = minibatch.get_minibatch(roidb, self.num_classes, self.mode)
        else:
            work_load_list = self.work_load_list
            ctx = self.ctx
            if work_load_list is None:
                work_load_list = [1] * len(ctx)
            assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
                "Invalid settings for work load. "
            slices = _split_input_slice(self.batch_size, work_load_list)

            data_list = []
            label_list = []
            for islice in slices:
                iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
                data, label = minibatch.get_minibatch(iroidb, self.num_classes, self.mode)
                data_list.append(data)
                label_list.append(label)

            # pad data first and then assign anchor (read label)
            data_tensor = tensor_vstack([batch['data'] for batch in data_list])
            for data, data_pad in zip(data_list, data_tensor):
                data['data'] = data_pad[np.newaxis, :]

            new_label_list = []
            for data, label in zip(data_list, label_list):
                # infer label shape
                data_shape = {k: v.shape for k, v in data.items()}
                del data_shape['im_info']
                _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
                feat_shape = [int(i) for i in feat_shape[0]]

                # assign anchor for label
                label = minibatch.assign_anchor(feat_shape, label['gt_boxes'], data['im_info'],
                                                self.feat_stride, self.anchor_scales,
                                                self.anchor_ratios, self.allowed_border)
                del data['im_info']
                new_label_list.append(label)

            all_data = dict()
            for key in ['data']:
                all_data[key] = tensor_vstack([batch[key] for batch in data_list])

            all_label = dict()
            all_label['label'] = tensor_vstack([batch['label'] for batch in new_label_list], pad=-1)
            for key in ['bbox_target', 'bbox_inside_weight', 'bbox_outside_weight']:
                all_label[key] = tensor_vstack([batch[key] for batch in new_label_list])

            self.data = [mx.nd.array(all_data['data'])]

            self.label = [mx.nd.array(all_label['label']),
                          mx.nd.array(all_label['bbox_target']),
                          mx.nd.array(all_label['bbox_inside_weight']),
                          mx.nd.array(all_label['bbox_outside_weight'])]
