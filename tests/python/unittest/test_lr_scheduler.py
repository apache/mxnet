import logging
import mxnet as mx 
import mxnet.optimizer as opt              

def test_lr_sceduler(lr, steps, lr_factor, warmup_step, warmup_lr):
    logging.basicConfig(level=logging.DEBUG) 

    lr_scheduler = None
    if warmup_step > 0 and warmup_lr > lr:
        lr_scheduler =  mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_factor, 
                    warmup_step = warmup_step, begin_lr=lr, stop_lr=warmup_lr)
    else:  
        lr_scheduler =  mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_factor) 

    optimizer_params = {
            'learning_rate': lr,
            'lr_scheduler': lr_scheduler}

    optimizer = opt.create('sgd', **optimizer_params)  
    updater = opt.get_updater(optimizer)     

    x = [[[[i*10+j for j in range(10)] for i in range(10)]]]
    x = mx.nd.array(x, dtype='float32')
    y = mx.nd.ones(shape = x.shape, dtype='float32') 

    res_lr = []
    for i in range(1,steps[-1] + 5):
        updater(0, y, x)
        cur_lr = optimizer._get_lr(0)
        res_lr.append(cur_lr)
        logging.info("step %d lr = %f", i, cur_lr)

    if warmup_step > 1:
        assert mx.test_utils.almost_equal(res_lr[warmup_step], warmup_lr, 1e-10) 
        lr = warmup_lr
    for i in range(len(steps)):
        assert mx.test_utils.almost_equal(res_lr[steps[i]], lr * pow(lr_factor, i + 1), 1e-10)    

if __name__ == "__main__":
    #Legal input
    test_lr_sceduler(lr = 0.2, steps = [8,12], lr_factor = 0.1, warmup_step = 0, warmup_lr = 0.1)
    test_lr_sceduler(lr = 0.02, steps = [8,12], lr_factor = 0.1, warmup_step = 1, warmup_lr = 0.1)
    test_lr_sceduler(lr = 0.02, steps = [8,12], lr_factor = 0.3, warmup_step = 5, warmup_lr = 0.1)
    test_lr_sceduler(lr = 0.002, steps = [8,12], lr_factor = 0.1, warmup_step = 7, warmup_lr = 0.1)
    #Illegal input
    """
    #Schedule step must be greater than warmup_step
    test_lr_sceduler(lr = 0.02, steps = [8,12], lr_factor = 0.1, warmup_step = 10, warmup_lr = 0.1)
    #stop_lr must larger than begin_lr
    test_lr_sceduler(lr = 0.02, steps = [8,12], lr_factor = 0.1, warmup_step = 10, warmup_lr = 0.001)
    #Schedule step must be an list
    test_lr_sceduler(lr = 0.02, steps = 8, lr_factor = 0.1, warmup_step = 5, warmup_lr = 0.1)
    #Factor must be no more than 1 to make lr reduce
    test_lr_sceduler(lr = 0.02, steps = [8,12], lr_factor = 2, warmup_step = 5, warmup_lr = 0.1)
    #Schedule step must be an increasing integer list
    test_lr_sceduler(lr = 0.02, steps = [12,8], lr_factor = 0.1, warmup_step = 5, warmup_lr = 0.1)
    """
