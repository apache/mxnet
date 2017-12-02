<if(lr)>
base_lr = <lr>
<endif>
<if(momentum)>
momentum = <momentum>
<endif>
<if(wd)>
wd = <wd>
<endif>
<if(epsilon)>
epsilon = <epsilon>
<endif>

optimizer_params={'learning_rate':base_lr <if(momentum)>, 'momentum':momentum<endif><if(wd)>, 'wd':wd<endif><if(epsilon)>, 'epsilon':epsilon<endif>}
module.init_optimizer(optimizer='<opt_name>', optimizer_params=optimizer_params)
