lr_update_steps = [<steps;separator=", ">]
if(batch_num in lr_update_steps):
    lr = optimizer_params['learning_rate']
    lr *= gamma
    optimizer_params['learning_rate'] = lr
