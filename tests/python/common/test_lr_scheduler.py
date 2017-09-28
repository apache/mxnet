import logging
import argparse
import mxnet as mx 
import mxnet.optimizer as opt              

def test_lr_sceduler(args):
    logging.basicConfig(level=logging.DEBUG)
    steps = [int(l) for l in args.lr_steps.split(',')]    

    lr_scheduler = None
    if args.warmup_step > 0 and args.warmup_lr > args.lr:
        lr_scheduler =  mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor, 
                    warmup_step = args.warmup_step, begin_lr=args.lr, stop_lr=args.warmup_lr)
    else:  
        lr_scheduler =  mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor) 

    optimizer_params = {
            'learning_rate': args.lr,
            'lr_scheduler': lr_scheduler}

    optimizer = opt.create('sgd', **optimizer_params)  
    updater = opt.get_updater(optimizer)     

    x = [[[[i*10+j for j in range(10)] for i in range(10)]]]
    x = mx.nd.array(x, dtype='float32')
    y = mx.nd.ones(shape = x.shape, dtype='float32') 

    for i in range(1,steps[-1] + 5):
        updater(0, y, x)
        cur_lr = optimizer._get_lr(0)
        logging.info("step %d lr = %f", i, cur_lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.02,
                       help='initial learning rate')
    parser.add_argument('--lr-factor', type=float, default=1,
                       help='the ratio to reduce lr on each step')
    parser.add_argument('--lr-steps', type=str, default='15,18',
                       help='the steps to reduce the lr')
    parser.add_argument('--warmup-step', type=int, default=10,
                       help='changes the learning rate for first warmup_step updates')
    parser.add_argument('--warmup-lr', type=float, default=0.1,
                       help='the learning rate will warmup to warmup_lr')
    args = parser.parse_args()

    test_lr_sceduler(args)
