import numpy as np
import math

class RandSampler(object):
    """
    Random sampler base class, used for data augmentation

    Parameters:
    ----------
    max_trials : int
        maximum trials, if exceed this number, give up anyway
    max_sample : int
        maximum random crop samples to be generated
    """
    def __init__(self, max_trials, max_sample):
        assert max_trials > 0
        self.max_trials = int(max_trials)
        assert max_sample >= 0
        self.max_sample = int(max_sample)

    def sample(self, label):
        """
        Interface for calling sampling function

        Parameters:
        ----------
        label : numpy.array (n x 5 matrix)
            ground-truths

        Returns:
        ----------
        list of (crop_box, label) tuples, if failed, return empty list []
        """
        return NotImplementedError


class RandCropper(RandSampler):
    """
    Random cropping original images with various settings

    Parameters:
    ----------
    min_scale : float
        minimum crop scale, (0, 1]
    max_scale : float
        maximum crop scale, (0, 1], must larger than min_scale
    min_aspect_ratio : float
        minimum crop aspect ratio, (0, 1]
    max_aspect_ratio : float
        maximum crop aspect ratio, [1, inf)
    min_overlap : float
        hreshold of minimum overlap between a rand crop and any gt
    max_trials : int
        maximum trials, if exceed this number, give up anyway
    max_sample : int
        maximum random crop samples to be generated
    """
    def __init__(self, min_scale=1., max_scale=1.,
                 min_aspect_ratio=1., max_aspect_ratio=1.,
                 min_overlap=0., max_trials=50, max_sample=1):
        super(RandCropper, self).__init__(max_trials, max_sample)
        assert min_scale <= max_scale, "min_scale must <= max_scale"
        assert 0 < min_scale and min_scale <= 1, "min_scale must in (0, 1]"
        assert 0 < max_scale and max_scale <= 1, "max_scale must in (0, 1]"
        self.min_scale = min_scale
        self.max_scale = max_scale
        assert 0 < min_aspect_ratio and min_aspect_ratio <= 1, "min_ratio must in (0, 1]"
        assert 1 <= max_aspect_ratio , "max_ratio must >= 1"
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        assert 0 <= min_overlap and min_overlap <= 1, "min_overlap must in [0,1]"
        self.min_overlap = min_overlap

        self.config = {'gt_constraint' : 'center'}

    def sample(self, label):
        """
        generate random cropping boxes according to parameters
        if satifactory crops generated, apply to ground-truth as well

        Parameters:
        ----------
        label : numpy.array (n x 5 matrix)
            ground-truths

        Returns:
        ----------
        list of (crop_box, label) tuples, if failed, return empty list []
        """
        samples = []
        count = 0
        for trial in range(self.max_trials):
            if count >= self.max_sample:
                return samples
            scale = np.random.uniform(self.min_scale, self.max_scale)
            min_ratio = max(self.min_aspect_ratio, scale * scale)
            max_ratio = min(self.max_aspect_ratio, 1. / scale / scale)
            ratio = math.sqrt(np.random.uniform(min_ratio, max_ratio))
            width = scale * ratio
            height = scale / ratio
            left = np.random.uniform(0., 1 - width)
            top = np.random.uniform(0., 1 - height)
            rand_box = (left, top, left + width, top + height)
            valid_mask = np.where(label[:, 0] > -1)[0]
            gt = label[valid_mask, :]
            ious = self._check_satisfy(rand_box, gt)
            if ious is not None:
                # transform gt labels after crop, discard bad ones
                l, t, r, b = rand_box
                new_gt_boxes = []
                new_width = r - l
                new_height = b - t
                for i in range(valid_mask.size):
                    if ious[i] > 0:
                        xmin = max(0., (gt[i, 1] - l) / new_width)
                        ymin = max(0., (gt[i, 2] - t) / new_height)
                        xmax = min(1., (gt[i, 3] - l) / new_width)
                        ymax = min(1., (gt[i, 4] - t) / new_height)
                        new_gt_boxes.append([gt[i, 0], xmin, ymin, xmax, ymax])
                if not new_gt_boxes:
                    continue
                new_gt_boxes = np.array(new_gt_boxes)
                label = np.lib.pad(new_gt_boxes,
                    ((0, label.shape[0]-new_gt_boxes.shape[0]), (0,0)), \
                    'constant', constant_values=(-1, -1))
                samples.append((rand_box, label))
                count += 1
        return samples

    def _check_satisfy(self, rand_box, gt_boxes):
        """
        check if overlap with any gt box is larger than threshold
        """
        l, t, r, b = rand_box
        num_gt = gt_boxes.shape[0]
        ls = np.ones(num_gt) * l
        ts = np.ones(num_gt) * t
        rs = np.ones(num_gt) * r
        bs = np.ones(num_gt) * b
        mask = np.where(ls < gt_boxes[:, 1])[0]
        ls[mask] = gt_boxes[mask, 1]
        mask = np.where(ts < gt_boxes[:, 2])[0]
        ts[mask] = gt_boxes[mask, 2]
        mask = np.where(rs > gt_boxes[:, 3])[0]
        rs[mask] = gt_boxes[mask, 3]
        mask = np.where(bs > gt_boxes[:, 4])[0]
        bs[mask] = gt_boxes[mask, 4]
        w = rs - ls
        w[w < 0] = 0
        h = bs - ts
        h[h < 0] = 0
        inter_area = h * w
        union_area = np.ones(num_gt) * max(0, r - l) * max(0, b - t)
        union_area += (gt_boxes[:, 3] - gt_boxes[:, 1]) * (gt_boxes[:, 4] - gt_boxes[:, 2])
        union_area -= inter_area
        ious = inter_area / union_area
        ious[union_area <= 0] = 0
        max_iou = np.amax(ious)
        if max_iou < self.min_overlap:
            return None
        # check ground-truth constraint
        if self.config['gt_constraint'] == 'center':
            for i in range(ious.shape[0]):
                if ious[i] > 0:
                    gt_x = (gt_boxes[i, 1] + gt_boxes[i, 3]) / 2.0
                    gt_y = (gt_boxes[i, 2] + gt_boxes[i, 4]) / 2.0
                    if gt_x < l or gt_x > r or gt_y < t or gt_y > b:
                        return None
        elif self.config['gt_constraint'] == 'corner':
            for i in range(ious.shape[0]):
                if ious[i] > 0:
                    if gt_boxes[i, 1] < l or gt_boxes[i, 3] > r \
                        or gt_boxes[i, 2] < t or gt_boxes[i, 4] > b:
                        return None
        return ious


class RandPadder(RandSampler):
    """
    Random cropping original images with various settings

    Parameters:
    ----------
    min_scale : float
        minimum crop scale, [1, inf)
    max_scale : float
        maximum crop scale, [1, inf), must larger than min_scale
    min_aspect_ratio : float
        minimum crop aspect ratio, (0, 1]
    max_aspect_ratio : float
        maximum crop aspect ratio, [1, inf)
    min_gt_scale : float
        minimum ground-truth scale to be satisfied after padding,
        either width or height, [0, 1]
    max_trials : int
        maximum trials, if exceed this number, give up anyway
    max_sample : int
        maximum random crop samples to be generated
    """
    def __init__(self, min_scale=1., max_scale=1., min_aspect_ratio=1., \
                 max_aspect_ratio=1., min_gt_scale=.01, max_trials=50,
                 max_sample=1):
        super(RandPadder, self).__init__(max_trials, max_sample)
        assert min_scale <= max_scale, "min_scale must <= max_scale"
        assert min_scale >= 1, "min_scale must in (0, 1]"
        self.min_scale = min_scale
        self.max_scale = max_scale
        assert 0 < min_aspect_ratio and min_aspect_ratio <= 1, "min_ratio must in (0, 1]"
        assert 1 <= max_aspect_ratio , "max_ratio must >= 1"
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        assert 0 <= min_gt_scale and min_gt_scale <= 1, "min_gt_scale must in [0, 1]"
        self.min_gt_scale = min_gt_scale

    def sample(self, label):
        """
        generate random padding boxes according to parameters
        if satifactory padding generated, apply to ground-truth as well

        Parameters:
        ----------
        label : numpy.array (n x 5 matrix)
            ground-truths

        Returns:
        ----------
        list of (crop_box, label) tuples, if failed, return empty list []
        """
        samples = []
        count = 0
        for trial in range(self.max_trials):
            if count >= self.max_sample:
                return samples
            scale = np.random.uniform(self.min_scale, self.max_scale)
            min_ratio = max(self.min_aspect_ratio, scale * scale)
            max_ratio = min(self.max_aspect_ratio, 1. / scale / scale)
            ratio = math.sqrt(np.random.uniform(min_ratio, max_ratio))
            width = scale * ratio
            if width < 1:
                continue
            height = scale / ratio
            if height < 1:
                continue
            left = np.random.uniform(0., 1 - width)
            top = np.random.uniform(0., 1 - height)
            right = left + width
            bot = top + height
            rand_box = (left, top, right, bot)
            valid_mask = np.where(label[:, 0] > -1)[0]
            gt = label[valid_mask, :]
            new_gt_boxes = []
            for i in range(gt.shape[0]):
                xmin = (gt[i, 1] - left) / width
                ymin = (gt[i, 2] - top) / height
                xmax = (gt[i, 3] - left) / width
                ymax = (gt[i, 4] - top) / height
                new_size = min(xmax - xmin, ymax - ymin)
                if new_size < self.min_gt_scale:
                    new_gt_boxes = []
                    break
                new_gt_boxes.append([gt[i, 0], xmin, ymin, xmax, ymax])
            if not new_gt_boxes:
                continue
            new_gt_boxes = np.array(new_gt_boxes)
            label = np.lib.pad(new_gt_boxes,
                ((0, label.shape[0]-new_gt_boxes.shape[0]), (0,0)), \
                'constant', constant_values=(-1, -1))
            samples.append((rand_box, label))
            count += 1
        return samples
