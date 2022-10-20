# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import copy
import numpy as np
from collections import OrderedDict
from neural_compressor.strategy.strategy import TuneStrategy, strategy_registry

plot_operator_influence = False

def calc_approx_error(expected_tensor: np.ndarray, observed_tensor: np.ndarray) -> float:
    '''
    Calculating relative error for one tensor
    '''
    error = observed_tensor - expected_tensor
    absolute_error = np.abs(error)
    mean_absolute_error = absolute_error.mean()
    mean_expected_value = np.abs(expected_tensor).mean()
    error = mean_absolute_error / mean_expected_value
    return error


def get_approx_errors(expected_tensors, observed_tensors):
    '''
    Calculating relative error for multiple tensors: Dict[tensors_name: str, tensor: np.ndarray]
    '''
    errors = {}
    for node_name in observed_tensors.keys():
        expected_tensor = expected_tensors[node_name][node_name]
        observed_tensor = observed_tensors[node_name][node_name]
        errors[node_name] = calc_approx_error(expected_tensor, observed_tensor)
    return errors


@strategy_registry
class MyCustomTuneStrategy(TuneStrategy):
    '''INC Custom strategy definition'''
    def __init__(self, model, conf, q_dataloader, q_func=None,
                 eval_dataloader=None, eval_func=None, dicts=None, q_hooks=None):
        super().__init__(
            model,
            conf,
            q_dataloader,
            q_func,
            eval_dataloader,
            eval_func,
            dicts,
            q_hooks)


    def get_qtensors(self, quant_cfg, node_list):
        '''
        Generating quantized model based on configuration and capturing intermediate tensors
        '''
        qmodel = self.adaptor.quantize(quant_cfg, self.model, self.calib_dataloader)
        tensors = self.adaptor.inspect_tensor(qmodel, self.calib_dataloader, node_list, [1]) # 1 is a batch index
        return tensors['activation'][0] # we need to specify that we want activation (layer output) because INC stores also weight tensors
                                        # 0 is the first batch
    def next_tune_cfg(self):
        FALLBACK_DTYPE = 'fp32'

        # creating base configuration - all nodes are quantized and calibrated with minmax algorithm
        best_cfg = {}
        best_cfg['calib_iteration'] = int(self.calib_iter[0]) # number of batches for calibration
        best_cfg['calib_sampling_size'] = int(self.calib_sampling_size[0]) # number of samples for calibration (multiplicity of batch)
        nodes_cfg = OrderedDict()
        nodes_cfg_idx = {}
        for node_key, cfgs in self.opwise_tune_cfgs.items():
            for i, cfg in enumerate(cfgs):
                if cfg['activation']['algorithm'] == 'minmax':
                    nodes_cfg_idx[node_key] = i
                    break
            nodes_cfg[node_key] = cfg
        best_cfg['op'] = nodes_cfg

        yield best_cfg

        # If fully quantized model does not meet the requirements, we proceed to exclude some nodes

        # Collecting tensors from the original model - expected tensors
        node_list = [op_name for (op_name, op_type) in best_cfg['op'].keys()]
        f32_tensors = self.adaptor.inspect_tensor(self.model, self.calib_dataloader, node_list, [1])
        f32_tensors = f32_tensors['activation'][0]

        # Collecting tensors from the fully quantized model
        q_tensors = self.get_qtensors(best_cfg, node_list)
        approx_errors = get_approx_errors(f32_tensors, q_tensors)

        # best_cfg['op'] is an OrderedDict, which order of elements should correspond to their
        # order in the computational graph
        for node_key, cfg in best_cfg['op'].items():
            # Node's key in INC is its name + its operator
            node_name, node_op = node_key
            # Checking what configuration options are available for this particular node
            capabilities = self.opwise_tune_space[node_key]['activation']['dtype']
            # If a particular node can be excluded from quanrtization ('fp32' in capabilities)
            # and current error is bigger than threshold value, we check what accuracy improvement
            # would be achieved by this exclusion
            if FALLBACK_DTYPE in capabilities and approx_errors[node_name] > 0.06:
                original_dtype = cfg['activation']['dtype']
                cfg['activation']['dtype'] = FALLBACK_DTYPE # Exclude the node from quantization

                # Collecting tensors for a new configuration with the current node excluded
                q_tensors = self.get_qtensors(best_cfg, node_list)
                # Calculating errors for the new configuration
                new_approx_errors = get_approx_errors(f32_tensors, q_tensors)
                # Calculating error differences for every node in a model
                err_diffs = {}
                for tensor_node_name in new_approx_errors.keys():
                    diff = approx_errors[tensor_node_name] - new_approx_errors[tensor_node_name]
                    err_diffs[tensor_node_name] = diff
                err_diffs_arr = np.array(list(err_diffs.values()))

                # If the sum of errors on the following layers is greater than the threshold value we
                # keep the node excluded
                threshold_sum_error_layers = err_diffs_arr.size * 0.007
                if err_diffs_arr.sum() >= threshold_sum_error_layers:
                    before = approx_errors
                    after = approx_errors.copy()
                    after.update(new_approx_errors)
                    if plot_operator_influence:
                        import matplotlib.pyplot as plt
                        plt.figure()
                        plt.plot(before.values(), marker='o', markersize=2.5, label='Before')
                        plt.plot(after.values(), marker='o', markersize=2.5, label='After')
                        plt.ylabel('Relative error')
                        plt.xlabel('Layer')
                        plt.legend()
                        plt.savefig(f'{node_name}_error.png')

                    approx_errors.update(new_approx_errors)
                    nodes_cfg_idx.pop(node_key) # Mark node as not quantizable
                else:
                    cfg['activation']['dtype'] = original_dtype

        yield best_cfg

        # Choosing calibration algorithm (kl or minmax) for every node which was not excluded from quantization
        for cfg in self.bayesian_configurations(best_cfg, nodes_cfg_idx):
            yield cfg

    def bayesian_params_to_tune_configs(self, params):
        '''
        Creating configuration from params - changing configurations' indexes for real configurations
        '''
        node_cfgs = {}
        for node_key, configs in self.opwise_quant_cfgs.items():
            if node_key in params:
                value = int(params[node_key])
                value = min(value, len(configs) - 1)
                node_cfgs[node_key] = copy.deepcopy(configs[value])
        return node_cfgs

    def bayesian_configurations(self, cfg_base, params_base):
        from neural_compressor.strategy.bayesian import BayesianOptimization

        # For each node we specify the possible range of values (we treat them as a configurations' index)
        pbounds = {}
        for node_key, configs in self.opwise_quant_cfgs.items():
            if node_key in params_base and len(configs) > 1:
                pbounds[node_key] = (0, len(configs))

        cfg = copy.deepcopy(cfg_base)
        if len(pbounds) == 0: # if there is nothing to be optimized, we finish
            cfg['op'].update(self.bayesian_params_to_tune_configs(params_base))
            return

        bayes_opt = BayesianOptimization(pbounds=pbounds, random_seed=self.cfg.tuning.random_seed)
        bayes_opt._space.register(params_base, self.last_tune_result[0]) # registering the outcome of current configuration
        while True:
            # Generating next configuration
            params = bayes_opt.gen_next_params()
            cfg['op'].update(self.bayesian_params_to_tune_configs(params))
            yield cfg
            try:
                # Registering the outcome
                bayes_opt._space.register(params, self.last_tune_result[0])
            except KeyError:
                pass
