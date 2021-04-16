using MxNet.Interop;
using MxNet.Numpy;
using MxNet.Sym.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.ND.Numpy
{
    public partial class npx
    {
        private static dynamic _api_internal = new _npx_internals();

        public static ndarray relu(ndarray data)
        {
            return activation(data);
        }

        public static ndarray activation(ndarray data, string act_type = "relu")
        {
            return _api_internal.activation(data: data, act_type: act_type);
        }

        public static ndarray batch_norm(ndarray x, ndarray gamma, ndarray beta, ndarray running_mean, 
                                        ndarray running_var, float eps= 0.001f, float momentum= 0.9f, bool fix_gamma= true, 
                                        bool use_global_stats= false, bool output_mean_var= false, int axis= 1, bool cudnn_off= false,
                                        float? min_calib_range= null, float? max_calib_range= null)
        {
            return _api_internal.batch_norm(x: x, gamma: gamma, beta: beta, running_mean: running_mean, running_var: running_var, eps: eps, 
                                            momentum: momentum, fix_gamma: fix_gamma, use_global_stats: use_global_stats, output_mean_var: output_mean_var, 
                                            axis: axis, cudnn_off: cudnn_off, min_calib_range: min_calib_range, max_calib_range: max_calib_range);
        }

        public static ndarray convolution(ndarray data, ndarray weight, ndarray bias = null, int[] kernel= null, 
                                        int[] stride= null, int[] dilate= null, int[] pad= null, int num_filter= 1, int num_group= 1, 
                                        int workspace= 1024, bool no_bias= false, string cudnn_tune= null, bool cudnn_off= false, string layout= null)
        {
            return _api_internal.convolution(data: data, weight: weight, bias: bias, kernel: new Shape(kernel), stride: new Shape(stride),
                                            dilate: new Shape(dilate), pad: new Shape(pad), num_filter: num_filter, num_group: num_group,
                                            workspace: workspace, no_bias: no_bias, cudnn_tune: cudnn_tune, cudnn_off: cudnn_off, layout: layout);
        }

        public static ndarray dropout(ndarray data, float p= 0.5f, string mode= "training", Shape axes= null, bool cudnn_off= true)
        {
            return _api_internal.dropout(data: data, p: p, mode: mode, axes: axes, cudnn_off: cudnn_off);
        }

        public static ndarray embedding(ndarray data, ndarray weight, int input_dim, int output_dim, DType dtype= null, bool sparse_grad= false)
        {
            return _api_internal.dropout(data: data, weight: weight, input_dim: input_dim, output_dim: output_dim, dtype: dtype, sparse_grad: sparse_grad);
        }

        public static ndarray fully_connected(ndarray x, ndarray weight, ndarray bias, int num_hidden, bool no_bias= true, bool flatten= true)
        {
            return _api_internal.fully_connected(data: x, weight: weight, bias: bias, num_hidden: num_hidden, no_bias: no_bias, flatten: flatten);
        }

        public static ndarray layer_norm(ndarray data, ndarray gamma, ndarray beta, int axis= -1, float eps= 9.99999975e-06f, bool output_mean_var= false)
        {
            return _api_internal.layer_norm(data: data, gamma: gamma, beta: beta, axis: axis, eps: eps, output_mean_var: output_mean_var);
        }

        public static ndarray pooling(ndarray data, int[] kernel, int[] stride = null, int[] pad = null, string pool_type = "max",
                                    string pooling_convention = "valid", bool global_pool = false, bool cudnn_off = false,
                                    int? p_value = null, int? count_include_pad = null, string layout = null)
        {
            return _api_internal.pooling(data: data, kernel: new Shape(kernel), stride: new Shape(stride), pad: new Shape(pad),
                                        pool_type: pool_type, pooling_convention: pooling_convention, global_pool: global_pool,
                                        cudnn_off: cudnn_off, p_value: p_value, count_include_pad: count_include_pad, layout: layout);
        }

        public static ndarray rnn(ndarray data, ndarray parameters, ndarray state, ndarray state_cell= null, ndarray sequence_length= null, 
                                string mode= null, int? state_size= null, int? num_layers= null, bool bidirectional= false, 
                                bool state_outputs= false, float p= 0, bool use_sequence_length= false, int? projection_size= null,
                                double? lstm_state_clip_min= null, double? lstm_state_clip_max= null, double? lstm_state_clip_nan= null)
        {
            return _api_internal.rnn(data: data, parameters: parameters, state: state, state_cell: state_cell, sequence_length: sequence_length,
                                    mode: mode, state_size: state_size, num_layers: num_layers, bidirectional: bidirectional, state_outputs: state_outputs,
                                    p: p, use_sequence_length: use_sequence_length, projection_size: projection_size, lstm_state_clip_min: lstm_state_clip_min,
                                    lstm_state_clip_max: lstm_state_clip_max, lstm_state_clip_nan: lstm_state_clip_nan);
        }

        public static ndarray leaky_relu(ndarray data, ndarray gamma= null, string act_type= "leaky", float slope= 0.25f, float lower_bound= 0.125f, float upper_bound= 0.333999991f)
        {
            return _api_internal.leaky_relu(data: data, gamma: gamma, act_type: act_type, slope: slope, lower_bound: lower_bound, upper_bound: upper_bound);
        }

        public static ndarray multibox_detection(ndarray cls_prob, ndarray loc_pred, ndarray anchor, bool clip= false,
                                                float threshold= 0.00999999978f, int background_id= 0, float nms_threshold= 0.5f, 
                                                bool force_suppress= false, float[] variances= null, int nms_topk= -1)
        {
            return _api_internal.multibox_detection(cls_prob: cls_prob, loc_pred: loc_pred, anchor: anchor, clip: clip, 
                                                    threshold: threshold, background_id: background_id, nms_threshold: nms_threshold,
                                                    force_suppress: force_suppress, variances: variances, nms_topk: nms_topk);
        }

        public static ndarray multibox_prior(ndarray data, float[] sizes = null, float[] ratios = null, bool clip = false, float[] steps = null, float[] offsets = null)
        {
            return _api_internal.multibox_prior(data: data, sizes: sizes, ratios: ratios, clip: clip, clip: clip, steps: steps, offsets: offsets);
        }

        public static ndarray multibox_target(ndarray anchor, ndarray label, ndarray cls_pred, float overlap_threshold = 0.5f,
                                            float ignore_label = -1, float negative_mining_ratio = -1, float negative_mining_thresh = 0.5f,
                                            int minimum_negative_samples = 0, float[] variances = null)
        {
            return _api_internal.multibox_target(anchor: anchor, label: label, cls_pred: cls_pred, overlap_threshold: overlap_threshold,
                                                ignore_label: ignore_label, negative_mining_ratio: negative_mining_ratio, negative_mining_thresh: negative_mining_thresh,
                                                minimum_negative_samples: minimum_negative_samples, variances: variances);
        }

        public static ndarray roi_pooling(ndarray data, ndarray rois, int[] pooled_size, float spatial_scale)
        {
            return _api_internal.roi_pooling(data: data, rois: rois, pooled_size: new Shape(pooled_size), spatial_scale: spatial_scale);
        }

        public static ndarray smooth_l1(ndarray data, float scalar)
        {
            return _api_internal.smooth_l1(data: data, scalar: scalar);
        }

        public static ndarray sigmoid(ndarray data)
        {
            return activation(data, "sigmoid");
        }

        public static ndarray softmax(ndarray data, int axis = -1, ndarray length = null, double? temperature = null, bool use_length = false, DType dtype = null)
        {
            return _api_internal.softmax(data: data, axis: axis, length: length, temperature: temperature, use_length: use_length, dtype: dtype);
        }

        public static ndarray log_softmax(ndarray data, int axis = -1, ndarray length = null, double? temperature = null, bool use_length = false, DType dtype = null)
        {
            return _api_internal.log_softmax(data: data, axis: axis, length: length, temperature: temperature, use_length: use_length, dtype: dtype);
        }

        public static ndarray topk(ndarray data, int axis = -1, int k = -1, string ret_typ = "value", bool is_ascend = false, DType dtype = null)
        {
            return _api_internal.topk(data: data, axis: axis, k: k, ret_typ: ret_typ, is_ascend: is_ascend, dtype: dtype);
        }

        public static ndarray waitall()
        {
            return _api_internal.waitall();
        }

        public static NDArrayDict load(string file)
        {
            return Utils.load(file);
        }

        public static void save(string file, ndarray arr)
        {
            Utils.save(file, arr);
        }

        public static ndarray one_hot(ndarray data, long depth, double on_value = 1.0, double off_value = 0.0, DType dtype = null)
        {
            return _api_internal.one_hot(data: data, depth: depth, on_value: on_value, off_value: off_value, dtype: dtype);
        }

        public static ndarray pick(ndarray data, ndarray index, int axis= -1, string mode= "clip", bool keepdims= false)
        {
            return _api_internal.pick(data: data, index: index, axis: axis, mode: keepdims, dtype: keepdims);
        }

        public static ndarray reshape_like(ndarray lhs, ndarray rhs, int? lhs_begin = null, int? lhs_end = null, int? rhs_begin = null, int? rhs_end = null)
        {
            return _api_internal.reshape_like(lhs: lhs, rhs: rhs, lhs_begin: lhs_begin, lhs_end: lhs_end, rhs_begin: rhs_begin, rhs_end: rhs_end);
        }

        public static ndarray batch_flatten(ndarray data)
        {
            return _api_internal.batch_flatten(data: data);
        }

        public static ndarray batch_dot(ndarray lhs, ndarray rhs, bool transpose_a = false, bool transpose_b = false, string forward_stype = null)
        {
            return _api_internal.batch_dot(lhs: lhs, rhs: rhs, transpose_a: transpose_a, transpose_b: transpose_b, forward_stype: forward_stype);
        }

        public static ndarray rois(ndarray data)
        {
            return _api_internal.rois(data: data);
        }

        public static ndarray gamma(ndarray data, ndarray @out = null)
        {
            @out = _api_internal.gamma(data: data);
            return @out;
        }

        public static ndarray sequence_mask(ndarray data, ndarray sequence_length = null, bool use_sequence_length = false, float value = 0, int axis = 0)
        {
            return _api_internal.sequence_mask(lhs: data, rhs: sequence_length, use_sequence_length: use_sequence_length, value: value, axis: axis);
        }
    }
}
