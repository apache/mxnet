lr = optimizer_params['learning_rate']
lr = base_lr * ( 1/(1 + math.exp(-gamma * (batch_num - stepsize))))
optimizer_params['learning_rate'] = lr
