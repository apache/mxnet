<if(params_file)>
arg_params, aux_params = load_params('<params_file>')
module.init_params(initializer=mx.init.Xavier(), arg_params=arg_params, aux_params=aux_params,
                   allow_missing=True)
<else>
module.init_params(initializer=mx.init.Xavier())
<endif>
