import argparse,logging,os
import mxnet as mx
from symbol_resnet import resnet
from mxnet import profiler

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def multi_factor_scheduler(begin_epoch, epoch_size, step=[60, 75, 90], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None


def main():
    if args.data_type == "cifar10":
        args.aug_level = 1
        args.num_classes = 10
        # depth should be one of 110, 164, 1001,...,which is should fit (args.depth-2)%9 == 0
        if((args.depth-2)%9 == 0 and args.depth >= 164):
            per_unit = [(args.depth-2)/9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif((args.depth-2)%6 == 0 and args.depth < 164):
            per_unit = [(args.depth-2)/6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))
        units = per_unit*3
        symbol = resnet(units=units, num_stage=3, filter_list=filter_list, num_class=args.num_classes,
                        data_type="cifar10", bottle_neck = bottle_neck, bn_mom=args.bn_mom, workspace=args.workspace,
                        memonger=args.memonger)
    elif args.data_type == "imagenet":
        args.num_classes = 1000
        if args.depth == 18:
            units = [2, 2, 2, 2]
        elif args.depth == 34:
            units = [3, 4, 6, 3]
        elif args.depth == 50:
            units = [3, 4, 6, 3]
        elif args.depth == 101:
            units = [3, 4, 23, 3]
        elif args.depth == 152:
            units = [3, 8, 36, 3]
        elif args.depth == 200:
            units = [3, 24, 36, 3]
        elif args.depth == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))
        symbol = resnet(units=units, num_stage=4, filter_list=[64, 256, 512, 1024, 2048] if args.depth >=50
                        else [64, 64, 128, 256, 512], num_class=args.num_classes, data_type="imagenet", bottle_neck = True
                        if args.depth >= 50 else False, bn_mom=args.bn_mom, workspace=args.workspace,
                        memonger=args.memonger)
    else:
         raise ValueError("do not support {} yet".format(args.data_type))
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    epoch_size = max(int(args.num_examples / args.batch_size / kv.num_workers), 1)
    begin_epoch = args.model_load_epoch if args.model_load_epoch else 0
    if not os.path.exists("./model"):
        os.mkdir("./model")
    model_prefix = "model/resnet-{}-{}-{}".format(args.data_type, args.depth, kv.rank)
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    arg_params = None
    aux_params = None
    if args.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.model_load_epoch)
    if args.memonger:
        import memonger
        symbol = memonger.search_plan(symbol, data=(args.batch_size, 3, 32, 32) if args.data_type=="cifar10"
                                                    else (args.batch_size, 3, 224, 224))
    train = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "train.rec") if args.data_type == 'cifar10' else
                              os.path.join(args.data_dir, "train_256_q90.rec") if args.aug_level == 1
                              else os.path.join(args.data_dir, "train_480_q90.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3, 32, 32) if args.data_type=="cifar10" else (3, 224, 224),
        batch_size          = args.batch_size,
        pad                 = 4 if args.data_type == "cifar10" else 0,
        fill_value          = 127,  # only used when pad is valid
        rand_crop           = True,
        max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10
        min_random_scale    = 1.0 if args.data_type == "cifar10" else 1.0 if args.aug_level == 1 else 0.533,  # 256.0/480.0
        max_aspect_ratio    = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 0.25,
        random_h            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 36,  # 0.4*90
        random_s            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        random_l            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        max_rotate_angle    = 0 if args.aug_level <= 2 else 10,
        max_shear_ratio     = 0 if args.aug_level <= 2 else 0.1,
        rand_mirror         = True,
        shuffle             = True,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)
    train = mx.io.ResizeIter(train,10)
    val = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "val.rec") if args.data_type == 'cifar10' else
                              os.path.join(args.data_dir, "val_256_q90.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = (3, 32, 32) if args.data_type=="cifar10" else (3, 224, 224),
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)
    model = mx.model.FeedForward(
        ctx                 = devs,
        symbol              = symbol,
        arg_params          = arg_params,
        aux_params          = aux_params,
        num_epoch           = 10, #200 if args.data_type == "cifar10" else 120,
        begin_epoch         = begin_epoch,
        learning_rate       = args.lr,
        momentum            = args.mom,
        wd                  = args.wd,
        optimizer           = 'nag',
        # optimizer          = 'sgd',
        initializer         = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        lr_scheduler        = multi_factor_scheduler(begin_epoch, epoch_size, step=[120, 160], factor=0.1)
                             if args.data_type=='cifar10' else
                             multi_factor_scheduler(begin_epoch, epoch_size, step=[30, 60, 90], factor=0.1),
        )
    val = None
    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = ['acc'] if args.data_type=='cifar10' else
                             ['acc', mx.metric.create('top_k_accuracy', top_k = 5)],
        kvstore            = kv,
        batch_end_callback = mx.callback.Speedometer(args.batch_size, args.frequent),
        epoch_end_callback = checkpoint)
    # logging.info("top-1 and top-5 acc is {}".format(model.score(X = val,
    #               eval_metric = ['acc', mx.metric.create('top_k_accuracy', top_k = 5)])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training resnet-v2")
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str, default='./data/imagenet/', help='the input data directory')
    parser.add_argument('--data-type', type=str, default='imagenet', help='the dataset type')
    parser.add_argument('--list-dir', type=str, default='./',
                        help='the directory which contain the training list file')
    parser.add_argument('--lr', type=float, default=0.1, help='initialization learning reate')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--bn-mom', type=float, default=0.9, help='momentum for batch normlization')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--batch-size', type=int, default=256, help='the batch size')
    parser.add_argument('--workspace', type=int, default=512, help='memory space size(MB) used in convolution, if xpu '
                        ' memory is oom, then you can try smaller vale, such as --workspace 256')
    parser.add_argument('--depth', type=int, default=50, help='the depth of resnet')
    parser.add_argument('--num-classes', type=int, default=1000, help='the class number of your task')
    parser.add_argument('--aug-level', type=int, default=2, choices=[1, 2, 3],
                        help='level 1: use only random crop and random mirror\n'
                             'level 2: add scale/aspect/hsv augmentation based on level 1\n'
                             'level 3: add rotation/shear augmentation based on level 2')
    parser.add_argument('--num-examples', type=int, default=1281167, help='the number of training examples')
    parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
    parser.add_argument('--model-load-epoch', type=int, default=0,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--frequent', type=int, default=50, help='frequency of logging')
    parser.add_argument('--memonger', action='store_true', default=False,
                        help='true means using memonger to save momory, https://github.com/dmlc/mxnet-memonger')
    parser.add_argument('--retrain', action='store_true', default=False, help='true means continue training')
    args = parser.parse_args()
    logging.info(args)
    #profiler.profiler_set_state('run')
    main()
    #profiler.profiler_set_state('stop')
