lr = optimizer_params['learning_rate']
lr = math.pow(base_lr * (1 - batch_num/max_iter), power)
optimizer_params['learning_rate'] = lr
