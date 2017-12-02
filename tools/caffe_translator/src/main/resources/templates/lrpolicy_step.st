if(batch_num % stepsize == 0):
    lr = optimizer_params['learning_rate']
    lr *= gamma
    optimizer_params['learning_rate'] = lr
