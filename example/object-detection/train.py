"""Train Gluon Object-Detection models."""
import argparse
from trainer import train_ssd

def parse_args():
    parser = argparse.ArgumentParser(description='Train a gluon detection network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algorithm', dest='algorithm', type=str, default='ssd',
                        help='which network to use')
    parser.add_argument('--data-shape', dest=data_shape, type=str, default='512',
                        help='image data shape, can be int or tuple')
    parser.add_argument('--model', dest='model', type=str, default='resnet18_v1',
                        help='base network to use, choices are models from gluon model_zoo')
    parser.add_argument('--dataset', dest='dataset', type=str, default='pascal',
                        help='which dataset to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--resume', dest='resume', type=int, default=-1,
                        help='resume training from epoch n')
    parser.add_argument('--prefix', dest='prefix', type=str, help='new model prefix',
                        default=os.path.join(os.path.dirname(__file__), 'model', 'default'))
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    parser.add_argument('--end-epoch', dest='end_epoch', help='end epoch of training',
                        default=240, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.004,
                        help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0.0005,
                        help='weight decay')
    # parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
    #                     help='red mean value')
    # parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
    #                     help='green mean value')
    # parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
    #                     help='blue mean value')
    # parser.add_argument('--lr-steps', dest='lr_refactor_step', type=str, default='80, 160',
    #                     help='refactor learning rate at specified epochs')
    # parser.add_argument('--lr-factor', dest='lr_refactor_ratio', type=str, default=0.1,
    #                     help='ratio to refactor learning rate')
    # parser.add_argument('--freeze', dest='freeze_pattern', type=str, default="^(conv1_|conv2_).*",
    #                     help='freeze layer pattern')
    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                        help='Save training log to file')
    parser.add_argument('--seed', dest='seed', type=int, default=123,
                        help="Random seed.")
    parser.add_argument('--dev', type=int, default=0,
                        help="Turn on develop mode with verbose informations.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    # choose algorithm
    if args.algorithm.lower() == 'ssd':
        model = '_'.join(args.algorithm, args.data_shape, args.model)
        train_ssd.train_net(model, )
