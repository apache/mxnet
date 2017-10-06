base_lr = <model.base_lr>
momentum = <model.momentum>
wd = <model.wd>

optimizer_params={'learning_rate':base_lr, 'momentum':momentum, 'wd':wd}

module.init_optimizer(optimizer='sgd', optimizer_params=optimizer_params)
