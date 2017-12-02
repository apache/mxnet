lr = optimizer_params['learning_rate']
lr = base_lr * math.pow((1 + gamma * batch_num), -power)
optimizer_params['learning_rate'] = lr
