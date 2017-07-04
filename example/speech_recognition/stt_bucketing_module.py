import mxnet as mx


class STTBucketingModule(mx.mod.BucketingModule):

    def save_checkpoint(self, prefix, epoch, save_optimizer_states=False):
        symbol, data_names, label_names = self._sym_gen(self._default_bucket_key)
        symbol.save('%s-symbol.json' % prefix)
        param_name = '%s-%04d.params' % (prefix, epoch)
        self.save_params(param_name)
        if save_optimizer_states:
            state_name = '%s-%04d.states' % (prefix, epoch)
            self._curr_module.save_optimizer_states(state_name)