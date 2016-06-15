import logging
from load_data import load_test_roidb
from rcnn.data_iter import ROIIter
from rcnn.symbol import get_symbol_vgg_test
from load_model import load_param
from rcnn.detector import Detector
from rcnn.tester import pred_eval


def test_net(imageset, year, root_path, devkit_path, prefix, epoch, ctx):
    """
    wrapper for detector
    :param imageset: image set to test on
    :param year: year of image set
    :param root_path: 'data' folder path
    :param devkit_path: 'VOCdevkit' folder path
    :param prefix: new model prefix
    :param epoch: new model epoch
    :param ctx: context to evaluate in
    :return: None
    """
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # load testing data
    voc, roidb = load_test_roidb(imageset, year, root_path, devkit_path)
    test_data = ROIIter(roidb, batch_size=1, shuffle=False, mode='test')

    # load model
    args, auxs = load_param(prefix, epoch, convert=True, ctx=ctx)

    # load symbol
    sym = get_symbol_vgg_test()

    # detect
    detector = Detector(sym, ctx, args, auxs)
    pred_eval(detector, test_data, voc, vis=False)
