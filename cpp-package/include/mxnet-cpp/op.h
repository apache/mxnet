/*!
*  Copyright (c) 2016 by Contributors
* \file op.h
* \brief definition of all the operators
* \author Chuntao Hong, Xin Li
*/

#ifndef _MXNETOP_H
#define _MXNETOP_H

#include <string>
#include <vector>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/shape.h"
#include "mxnet-cpp/operator.h"
#include "dmlc/optional.h"

namespace mxnet {
namespace cpp {

/*!
 * \breif Batch normalization.
 *
 *        Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis:
 *
 *        .. math::
 *
 *        data\_mean[i] = mean(data[:,i,:,...]) \\
 *        data\_var[i] = var(data[:,i,:,...])
 *
 *        Then compute the normalized output, which has the same shape as input, as
 *
 *        .. math::
 *
 *        out[:,i,:,...] = \frac{data[:,i,:,...] -
 *
 *        Both *mean* and *var* returns a scalar by treating the input as a vector.
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        ``data_var`` as well, which are needed for the backward pass.
 *
 *        Besides the inputs and the outputs, this operator accepts two auxiliary
 *        states, ``moving_mean`` and ``moving_var``, which are *k*-length
 *        vectors. They are global statistics for the whole dataset, which are updated
 *        by::
 *
 *        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
 *        moving_var = moving_var * momentum + data_var * (1 - momentum)
 *
 *        If ``use_global_stats`` is set to be true, then ``moving_mean`` and
 *        ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
 *        the output. It is often used during inference.
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is
 *        then set ``gamma`` to 1 and its gradient to 0.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/batch_norm.cc:L84
 * \param symbol_name name of the resulting symbol
 * \param data Input data to batch normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param eps Epsilon to prevent div 0
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of local
 * \param output_mean_var Output All,normal mean and var
 * \return new symbol
 */
inline Symbol BatchNorm(const std::string& symbol_name,
                        Symbol data,
                        Symbol gamma,
                        Symbol beta,
                        mx_float eps = 0.001,
                        mx_float momentum = 0.9,
                        bool fix_gamma = true,
                        bool use_global_stats = false,
                        bool output_mean_var = false) {
  return Operator("BatchNorm")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetParam("output_mean_var", output_mean_var)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol(symbol_name);
}

/*! \breif Activation function to be applied.
 */
enum class LeakyReLUActType {
  elu = 0,
  leaky = 1,
  prelu = 2,
  rrelu = 3
};

/*!
 * \breif Leaky ReLu activation
 *
 *        The following types are supported:
 *
 *        - *elu*: ``y = x > 0 ? x : slop * (exp(x)-1)``
 *        - *leaky*: ``y = x > 0 ? x : slope * x``
 *        - *prelu*: same as *leaky* but the ``slope`` is learnable.
 *        - *rrelu*: same as *leaky* but the ``slope`` is uniformly randomly chosen from
 *        *[lower_bound, upper_bound)* for training, while fixed to be
 *        *(lower_bound+upper_bound)/2* for inference.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/leaky_relu.cc:L36
 * \param symbol_name name of the resulting symbol
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \param slope Init slope for the activation. (For leaky and elu only)
 * \param lower_bound Lower bound of random slope. (For rrelu only)
 * \param upper_bound Upper bound of random slope. (For rrelu only)
 * \return new symbol
 */
inline Symbol LeakyReLU(const std::string& symbol_name,
                        Symbol data,
                        LeakyReLUActType act_type = LeakyReLUActType::leaky,
                        mx_float slope = 0.25,
                        mx_float lower_bound = 0.125,
                        mx_float upper_bound = 0.334) {
  static const char *LeakyReLUActTypeValues[] = {
    "elu",
    "leaky",
    "prelu",
    "rrelu"
  };
  return Operator("LeakyReLU")
           .SetParam("act_type", LeakyReLUActTypeValues[int(act_type)])
           .SetParam("slope", slope)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Concate a list of array along a given axis.
 *
 *        The dimension sizes of the input arrays on the given axis should be the same.
 *
 *        For example::
 *
 *        x = [[1,1],[1,1]]
 *        y = [[2,2],[2,2]]
 *        z = [[3,3],[3,3],[3,3]]
 *
 *        Concat(x,y,z,dim=0) = [[ 1.,  1.],
 *        [ 1.,  1.],
 *        [ 2.,  2.],
 *        [ 2.,  2.],
 *        [ 3.,  3.],
 *        [ 3.,  3.],
 *        [ 3.,  3.]]
 *
 *        Concat(x,y,z,dim=1) = [[ 1.,  1.,  2.,  2.],
 *        [ 1.,  1.,  2.,  2.]]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/concat.cc:L69
 * \param symbol_name name of the resulting symbol
 * \param data List of tensors to concatenate
 * \param num_args Number of inputs to be concated.
 * \param dim the dimension to be concated.
 * \return new symbol
 */
inline Symbol Concat(const std::string& symbol_name,
                     const std::vector<Symbol>& data,
                     int num_args,
                     int dim = 1) {
  return Operator("Concat")
           .SetParam("num_args", num_args)
           .SetParam("dim", dim)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply a sparse regularization to the output a sigmoid activation function.
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param sparseness_target The sparseness target
 * \param penalty The tradeoff parameter for the sparseness penalty
 * \param momentum The momentum for running average
 * \return new symbol
 */
inline Symbol IdentityAttachKLSparseReg(const std::string& symbol_name,
                                        Symbol data,
                                        mx_float sparseness_target = 0.1,
                                        mx_float penalty = 0.001,
                                        mx_float momentum = 0.9) {
  return Operator("IdentityAttachKLSparseReg")
           .SetParam("sparseness_target", sparseness_target)
           .SetParam("penalty", penalty)
           .SetParam("momentum", momentum)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate cross_entropy(data, one_hot(label))
 *
 *        From:/home/xlidc/mxnet/src/operator/loss_binary_op.cc:12
 * \param symbol_name name of the resulting symbol
 * \param data Input data
 * \param label Input label
 * \return new symbol
 */
inline Symbol softmax_cross_entropy(const std::string& symbol_name,
                                    Symbol data,
                                    Symbol label) {
  return Operator("softmax_cross_entropy")
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*! \breif Padding type to use. "constant" pads all values with a constant value, the
 *        value of which can be specified with the constant_value option. "edge" uses the
 */
enum class PadMode {
  constant = 0,
  edge = 1
};

/*!
 * \breif Pad an array.
 *
 *        Only supports 4-D and 5-D input array.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/pad.cc:L407
 * \param symbol_name name of the resulting symbol
 * \param data An n-dimensional input tensor.
 * \param mode Padding type to use. "constant" pads all values with a constant value, the
 *        value of which can be specified with the constant_value option. "edge" uses the
 * \param pad_width A tuple of padding widths of length 2*r, where r is the rank of the
 *        input tensor, specifying number of values padded to the edges of each axis.
 *        (before_1, after_1, ... , before_N, after_N) unique pad widths for each axis.
 * \param constant_value This option is only used when mode is "constant". This value
 * \return new symbol
 */
inline Symbol Pad(const std::string& symbol_name,
                  Symbol data,
                  PadMode mode,
                  Shape pad_width,
                  double constant_value = 0) {
  static const char *PadModeValues[] = {
    "constant",
    "edge"
  };
  return Operator("Pad")
           .SetParam("mode", PadModeValues[int(mode)])
           .SetParam("pad_width", pad_width)
           .SetParam("constant_value", constant_value)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Updater function for sgd optimizer
 * \param symbol_name name of the resulting symbol
 * \param lr learning_rate
 * \param wd weight decay
 * \param rescale_grad rescale gradient as grad = rescale_grad*grad.
 * \param clip_gradient If greater than 0, clip gradient to grad = max(min(grad,
 * \return new symbol
 */
inline Symbol sgd_update(const std::string& symbol_name,
                         mx_float lr,
                         mx_float wd = 0,
                         mx_float rescale_grad = 1,
                         mx_float clip_gradient = -1) {
  return Operator("sgd_update")
           .SetParam("lr", lr)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Updater function for sgd optimizer
 * \param symbol_name name of the resulting symbol
 * \param lr learning_rate
 * \param momentum momentum
 * \param wd weight decay
 * \param rescale_grad rescale gradient as grad = rescale_grad*grad.
 * \param clip_gradient If greater than 0, clip gradient to grad = max(min(grad,
 * \return new symbol
 */
inline Symbol sgd_mom_update(const std::string& symbol_name,
                             mx_float lr,
                             mx_float momentum = 0,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1) {
  return Operator("sgd_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Updater function for adam optimizer
 * \param symbol_name name of the resulting symbol
 * \param lr learning_rate
 * \param beta1 beta1
 * \param beta2 beta2
 * \param epsilon epsilon
 * \param wd weight decay
 * \param rescale_grad rescale gradient as grad = rescale_grad*grad.
 * \param clip_gradient If greater than 0, clip gradient to grad = max(min(grad,
 * \return new symbol
 */
inline Symbol adam_update(const std::string& symbol_name,
                          mx_float lr,
                          mx_float beta1 = 0.9,
                          mx_float beta2 = 0.999,
                          mx_float epsilon = 1e-08,
                          mx_float wd = 0,
                          mx_float rescale_grad = 1,
                          mx_float clip_gradient = -1) {
  return Operator("adam_update")
           .SetParam("lr", lr)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Updater function for RMSProp optimizer. The RMSProp code follows the version in
 *        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 * \param symbol_name name of the resulting symbol
 * \param lr learning_rate
 * \param gamma1 gamma1
 * \param epsilon epsilon
 * \param wd weight decay
 * \param rescale_grad rescale gradient as grad = rescale_grad*grad.
 * \param clip_gradient If greater than 0, clip gradient to grad = max(min(grad,
 * \param clip_weights If greater than 0, clip weights to weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmsprop_update(const std::string& symbol_name,
                             mx_float lr,
                             mx_float gamma1 = 0.95,
                             mx_float epsilon = 1e-08,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1,
                             mx_float clip_weights = -1) {
  return Operator("rmsprop_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Updater function for RMSPropAlex optimizer. The RMSPropAlex code follows the
 *        version in http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves,
 * \param symbol_name name of the resulting symbol
 * \param lr learning_rate
 * \param gamma1 gamma1
 * \param gamma2 gamma2
 * \param epsilon epsilon
 * \param wd weight decay
 * \param rescale_grad rescale gradient as grad = rescale_grad*grad.
 * \param clip_gradient If greater than 0, clip gradient to grad = max(min(grad,
 * \param clip_weights If greater than 0, clip weights to weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmspropalex_update(const std::string& symbol_name,
                                 mx_float lr,
                                 mx_float gamma1 = 0.95,
                                 mx_float gamma2 = 0.9,
                                 mx_float epsilon = 1e-08,
                                 mx_float wd = 0,
                                 mx_float rescale_grad = 1,
                                 mx_float clip_gradient = -1,
                                 mx_float clip_weights = -1) {
  return Operator("rmspropalex_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("gamma2", gamma2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Interchange two axes of an array.
 *
 *        Examples::
 *
 *        x = [[1, 2, 3]])
 *        swapaxes(x, 0, 1) = [[ 1],
 *        [ 2],
 *        [ 3]]
 *
 *        x = [[[ 0, 1],
 *        [ 2, 3]],
 *        [[ 4, 5],
 *        [ 6, 7]]]  // (2,2,2) array
 *
 *        swapaxes(x, 0, 2) = [[[ 0, 4],
 *        [ 2, 6]],
 *        [[ 1, 5],
 *        [ 3, 7]]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/swapaxis.cc:L55
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \param dim1 the first axis to be swapped.
 * \param dim2 the second axis to be swapped.
 * \return new symbol
 */
inline Symbol SwapAxis(const std::string& symbol_name,
                       Symbol data,
                       uint32_t dim1 = 0,
                       uint32_t dim2 = 0) {
  return Operator("SwapAxis")
           .SetParam("dim1", dim1)
           .SetParam("dim2", dim2)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Split an array along a particular axis into multiple sub-arrays.
 *
 *        Assume the input array has shape ``(d_0, ..., d_n)`` and we slice it into *m*
 *        (``num_outputs=m``) subarrays along axis *k*, then we will obtain a list of *m*
 *        arrays with each of which has shape ``(d_0, ..., d_k/m, ..., d_n)``.
 *
 *        For example::
 *
 *        x = [[1, 2],
 *        [3, 4],
 *        [5, 6],
 *        [7, 8]]  // 4x2 array
 *
 *        y = split(x, axis=0, num_outputs=4) // a list of 4 arrays
 *        y[0] = [[ 1.,  2.]]  // 1x2 array
 *
 *        z = split(x, axis=0, num_outputs=2) // a list of 2 arrays
 *        z[0] = [[ 1.,  2.],
 *        [ 3.,  4.]]
 *
 *        When setting optional argument ``squeeze_axis=1``, then the *k*-dimension will
 *        be removed from the shape if it becomes 1::
 *
 *        y = split(x, axis=0, num_outputs=4, squeeze_axis=1)
 *        y[0] = [ 1.,  2.]  // (2,) vector
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/slice_channel.cc:L50
 * \param symbol_name name of the resulting symbol
 * \param num_outputs Number of outputs to be sliced.
 * \param axis Dimension along which to slice.
 * \param squeeze_axis If true, the dimension will be squeezed. Also, input.shape[axis]
 * \return new symbol
 */
inline Symbol SliceChannel(const std::string& symbol_name,
                           int num_outputs,
                           int axis = 1,
                           bool squeeze_axis = false) {
  return Operator("SliceChannel")
           .SetParam("num_outputs", num_outputs)
           .SetParam("axis", axis)
           .SetParam("squeeze_axis", squeeze_axis)
           .CreateSymbol(symbol_name);
}

/*! \breif upsampling method
 */
enum class UpSamplingSampleType {
  bilinear = 0,
  nearest = 1
};

/*! \breif How to handle multiple input. concat means concatenate upsampled images along
 *        the channel dimension. sum means add all images together, only available for
 */
enum class UpSamplingMultiInputMode {
  concat = 0,
  sum = 1
};

/*!
 * \breif Perform nearest neighboor/bilinear up sampling to inputs
 * \param symbol_name name of the resulting symbol
 * \param data Array of tensors to upsample
 * \param scale Up sampling scale
 * \param sample_type upsampling method
 * \param num_args Number of inputs to be upsampled. For nearest neighbor upsampling,
 *        this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other
 *        inputs will be upsampled to thesame size. For bilinear upsampling this must be
 * \param num_filter Input filter. Only used by bilinear sample_type.
 * \param multi_input_mode How to handle multiple input. concat means concatenate
 *        upsampled images along the channel dimension. sum means add all images
 * \param workspace Tmp workspace for deconvolution (MB)
 * \return new symbol
 */
inline Symbol UpSampling(const std::string& symbol_name,
                         const std::vector<Symbol>& data,
                         uint32_t scale,
                         UpSamplingSampleType sample_type,
                         int num_args,
                         uint32_t num_filter = 0,
                         UpSamplingMultiInputMode multi_input_mode = UpSamplingMultiInputMode::concat,
                         uint64_t workspace = 512) {
  static const char *UpSamplingSampleTypeValues[] = {
    "bilinear",
    "nearest"
  };
  static const char *UpSamplingMultiInputModeValues[] = {
    "concat",
    "sum"
  };
  return Operator("UpSampling")
           .SetParam("scale", scale)
           .SetParam("sample_type", UpSamplingSampleTypeValues[int(sample_type)])
           .SetParam("num_args", num_args)
           .SetParam("num_filter", num_filter)
           .SetParam("multi_input_mode", UpSamplingMultiInputModeValues[int(multi_input_mode)])
           .SetParam("workspace", workspace)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_add(const std::string& symbol_name,
                           Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate Smooth L1 Loss(lhs, scalar)
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/elemwise_binary_scalar_op_extended.cc:63
 * \param symbol_name name of the resulting symbol
 * \param data source input
 * \param scalar scalar input
 * \return new symbol
 */
inline Symbol smooth_l1(const std::string& symbol_name,
                        Symbol data,
                        mx_float scalar) {
  return Operator("smooth_l1")
           .SetParam("scalar", scalar)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif The return type. "value" means returning the top k values, "indices" means
 *        returning the indices of the top k values, "mask" means to return a mask array
 *        containing 0 and 1. 1 means the top k values. "both" means to return both value
 */
enum class TopkRetTyp {
  both = 0,
  indices = 1,
  mask = 2,
  value = 3
};

/*!
 * \breif Return the top *k* elements in an array.
 *
 *        Examples::
 *
 *        x = [[ 0.3,  0.2,  0.4],
 *        [ 0.1,  0.3,  0.2]]
 *
 *        // return the index of the largest element on last axis
 *        topk(x) = [[ 2.],
 *        [ 1.]]
 *
 *        // return the value of the top-2 elements on last axis
 *        topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],
 *        [ 0.3,  0.2]]
 *
 *        // flatten and then return both index and value
 *        topk(x, ret_typ='both', k=2, axis=None) = [ 0.4,  0.3], [ 2.,  0.]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/ordering_op.cc:L36
 * \param symbol_name name of the resulting symbol
 * \param src Source input
 * \param axis Axis along which to choose the top k indices. If not given, the flattened
 * \param k Number of top elements to select, should be always smaller than or equal to
 * \param ret_typ The return type. "value" means returning the top k values, "indices"
 *        means returning the indices of the top k values, "mask" means to return a mask
 *        array containing 0 and 1. 1 means the top k values. "both" means to return both
 * \param is_ascend Whether to choose k largest or k smallest. Top K largest elements
 * \return new symbol
 */
inline Symbol topk(const std::string& symbol_name,
                   Symbol src,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   int k = 1,
                   TopkRetTyp ret_typ = TopkRetTyp::indices,
                   bool is_ascend = false) {
  static const char *TopkRetTypValues[] = {
    "both",
    "indices",
    "mask",
    "value"
  };
  return Operator("topk")
           .SetParam("axis", axis)
           .SetParam("k", k)
           .SetParam("ret_typ", TopkRetTypValues[int(ret_typ)])
           .SetParam("is_ascend", is_ascend)
           .SetInput("src", src)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Return a sorted copy of an array.
 *
 *        Examples::
 *
 *        x = [[ 1, 4],
 *        [ 3, 1]]
 *
 *        // sort along the last axis
 *        sort(x) = [[ 1.,  4.],
 *        [ 1.,  3.]]
 *
 *        // flatten and then sort
 *        sort(x, axis=None) = [ 1.,  1.,  3.,  4.]
 *
 *        // sort long the first axis
 *        sort(x, axis=0) = [[ 1.,  1.],
 *        [ 3.,  4.]]
 *
 *        // in a descend order
 *        sort(x, is_ascend=0) = [[ 4.,  1.],
 *        [ 3.,  1.]]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/ordering_op.cc:L99
 * \param symbol_name name of the resulting symbol
 * \param src Source input
 * \param axis Axis along which to choose sort the input tensor. If not given, the
 * \param is_ascend Whether sort in ascending or descending order.
 * \return new symbol
 */
inline Symbol sort(const std::string& symbol_name,
                   Symbol src,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   bool is_ascend = true) {
  return Operator("sort")
           .SetParam("axis", axis)
           .SetParam("is_ascend", is_ascend)
           .SetInput("src", src)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the indices that can sort an array.
 *
 *        Examples::
 *
 *        x = [[ 0.3,  0.2,  0.4],
 *        [ 0.1,  0.3,  0.2]]
 *
 *        // sort along axis -1
 *        argsort(x) = [[ 1.,  0.,  2.],
 *        [ 0.,  2.,  1.]]
 *
 *        // sort along axis 0
 *        argsort(x, axis=0) = [[ 1.,  0.,  1.]
 *        [ 0.,  1.,  0.]]
 *
 *        // flatten and then sort
 *        argsort(x, axis=None) = [ 3.,  1.,  5.,  0.,  4.,  2.]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/ordering_op.cc:L146
 * \param symbol_name name of the resulting symbol
 * \param src Source input
 * \param axis Axis along which to sort the input tensor. If not given, the flattened
 * \param is_ascend Whether sort in ascending or descending order.
 * \return new symbol
 */
inline Symbol argsort(const std::string& symbol_name,
                      Symbol src,
                      dmlc::optional<int> axis = dmlc::optional<int>(-1),
                      bool is_ascend = true) {
  return Operator("argsort")
           .SetParam("axis", axis)
           .SetParam("is_ascend", is_ascend)
           .SetInput("src", src)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Get output from a symbol and pass 0 gradient back
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:31
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol BlockGrad(const std::string& symbol_name,
                        Symbol data) {
  return Operator("BlockGrad")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Output data type.
 */
enum class CastDtype {
  float16 = 0,
  float32 = 1,
  float64 = 2,
  int32 = 3,
  uint8 = 4
};

/*!
 * \breif Cast to a specified type, element-wise.
 *
 *        For example::
 *
 *        cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
 *        cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L65
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param dtype Output data type.
 * \return new symbol
 */
inline Symbol Cast(const std::string& symbol_name,
                   Symbol data,
                   CastDtype dtype) {
  static const char *CastDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("Cast")
           .SetParam("dtype", CastDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Negate src
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:84
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol negative(const std::string& symbol_name,
                       Symbol data) {
  return Operator("negative")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the absolute value of array elements, element-wise.
 *
 *        For example:
 *        abs([-2, 0, 3]) = [2, 0, 3]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L95
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol abs(const std::string& symbol_name,
                  Symbol data) {
  return Operator("abs")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the indication sign of array elements, element-wise.
 *
 *        For example::
 *        sign([-2, 0, 3]) = [-1, 0, 1]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L109
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol sign(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sign")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Round elements of the array to the nearest integer, element-wise.
 *
 *        For example::
 *        round([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -2.,  2.,  2.,  2.]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L122
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol round(const std::string& symbol_name,
                    Symbol data) {
  return Operator("round")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Return the ceiling of the input, element-wise.
 *
 *        For example::
 *        ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L132
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol ceil(const std::string& symbol_name,
                   Symbol data) {
  return Operator("ceil")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Return the floor of the input, element-wise.
 *
 *        For example::
 *        floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L141
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol floor(const std::string& symbol_name,
                    Symbol data) {
  return Operator("floor")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Round elements of the array to the nearest integer, element-wise.
 *
 *        For example::
 *        rint([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -2.,  1.,  2.,  2.]
 *
 *        The difference to ``round`` is that ``rint`` returns ``n`` for input ``n.5``
 *        while ``round`` returns ``n+1`` for ``n>=0``.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L154
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol rint(const std::string& symbol_name,
                   Symbol data) {
  return Operator("rint")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Round elements of the array to the nearest integer towards
 *        zero, element-wise.
 *
 *        For example::
 *        fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L164
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol fix(const std::string& symbol_name,
                  Symbol data) {
  return Operator("fix")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate the square of an array, element-wise.
 *
 *        For example::
 *        square(x) = x^2
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L174
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol square(const std::string& symbol_name,
                     Symbol data) {
  return Operator("square")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate the square-root of an array, element-wise.
 *
 *        For example::
 *        sqrt(x) = \sqrt{x}
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L187
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol sqrt(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sqrt")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate the inverse square-root of an array, element-wise.
 *
 *        For example::
 *        rsqrt(x) = 1/\sqrt{x}
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L200
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol rsqrt(const std::string& symbol_name,
                    Symbol data) {
  return Operator("rsqrt")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate the exponential of the array, element-wise
 *
 *        For example::
 *        exp(x) = e^x \approx 2.718^x
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L215
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol exp(const std::string& symbol_name,
                  Symbol data) {
  return Operator("exp")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Natural logarithm, element-wise.
 *
 *        The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L225
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol log(const std::string& symbol_name,
                  Symbol data) {
  return Operator("log")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate the base 10 logarithm of the array, element-wise.
 *
 *        ``10**log10(x) = x``
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L235
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol log10(const std::string& symbol_name,
                    Symbol data) {
  return Operator("log10")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate the base 2 logarithm of the array, element-wise.
 *
 *        ``2**log2(x) = x``
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L245
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol log2(const std::string& symbol_name,
                   Symbol data) {
  return Operator("log2")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Trigonometric sine, element-wise.
 *
 *        Then input is in radians (:math:`2\pi` rad equals 360 degress).
 *
 *        .. math::
 *        sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L261
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol sin(const std::string& symbol_name,
                  Symbol data) {
  return Operator("sin")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate ``log(1 + x)``
 *
 *        This function is more accurate than ``log(1 + x)``  for small ``x`` so that
 *        :math:`1+x\approx 1`
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L275
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol log1p(const std::string& symbol_name,
                    Symbol data) {
  return Operator("log1p")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate ``exp(x) - 1``
 *
 *        This function provides greater precision than ``exp(x) - 1`` for small values
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L288
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol expm1(const std::string& symbol_name,
                    Symbol data) {
  return Operator("expm1")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Cosine, element-wise.
 *
 *        Then input is in radians (:math:`2\pi` rad equals 360 degress).
 *
 *        .. math::
 *        cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L304
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol cos(const std::string& symbol_name,
                  Symbol data) {
  return Operator("cos")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Tangent, element-wise.
 *
 *        Then input is in radians (:math:`2\pi` rad equals 360 degress).
 *
 *        .. math::
 *        tan([0, \pi/4, \pi/2]) = [0, 1, -inf]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L320
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol tan(const std::string& symbol_name,
                  Symbol data) {
  return Operator("tan")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Inverse sine, element-wise.
 *
 *        The input should be in range :math:`[-1, 1]`.
 *        The output is in the closed interval :math:`[-\pi/2, \pi/2]`
 *
 *        .. math::
 *        arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L337
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol arcsin(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arcsin")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Inverse cosine, element-wise.
 *
 *        The input should be in range :math:`[-1, 1]`.
 *        The output is in the closed interval :math:`[0, \pi]`
 *
 *        .. math::
 *        arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L354
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol arccos(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arccos")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Inverse tangent, element-wise.
 *
 *        The output is in the closed interval :math:`[-\pi/2, \pi/2]`
 *
 *        .. math::
 *        arccos([-1, 0, 1]) = [-\pi/4, 0, \pi/4]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L370
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol arctan(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arctan")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Convert angles from radians to degrees.
 *
 *        .. math::
 *        degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L384
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol degrees(const std::string& symbol_name,
                      Symbol data) {
  return Operator("degrees")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Convert angles from degrees to radians.
 *
 *        .. math::
 *        radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L398
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol radians(const std::string& symbol_name,
                      Symbol data) {
  return Operator("radians")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Hyperbolic sine, element-wise.
 *
 *        For example::
 *        sinh(x) = 0.5\times(exp(x) - exp(-x))
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L412
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol sinh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sinh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Hyperbolic cosine, element-wise.
 *
 *        For example::
 *        cosh(x) = 0.5\times(exp(x) + exp(-x))
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L426
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol cosh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("cosh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Hyperbolic tangent element-wise.
 *
 *        For example::
 *        tanh(x) = sinh(x) / cosh(x)
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L440
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol tanh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("tanh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Inverse hyperbolic sine, element-wise.
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L450
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol arcsinh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arcsinh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Inverse hyperbolic cosine, element-wise.
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L460
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol arccosh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arccosh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Inverse hyperbolic tangent, element-wise.
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L470
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol arctanh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arctanh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif The gamma function (extension of the factorial function), element-wise
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:479
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol gamma(const std::string& symbol_name,
                    Symbol data) {
  return Operator("gamma")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Log of the absolute value of the gamma function, element-wise
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:488
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol gammaln(const std::string& symbol_name,
                      Symbol data) {
  return Operator("gammaln")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Map integer index to vector representations (embeddings). Those embeddings are
 *        learnable parameters. For a input of shape (d1, ..., dK), the output shape is
 *        (d1, ..., dK, output_dim). All the input values should be integers in the range
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/indexing_op.cc:19
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the EmbeddingOp.
 * \param weight Embedding weight matrix.
 * \param input_dim vocabulary size of the input indices.
 * \param output_dim dimension of the embedding vectors.
 * \return new symbol
 */
inline Symbol Embedding(const std::string& symbol_name,
                        Symbol data,
                        Symbol weight,
                        int input_dim,
                        int output_dim) {
  return Operator("Embedding")
           .SetParam("input_dim", input_dim)
           .SetParam("output_dim", output_dim)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .CreateSymbol(symbol_name);
}

/*! \breif specify how out-of-bound indices bahave.
 */
enum class TakeMode {
  clip = 0,
  raise = 1,
  wrap = 2
};

/*!
 * \breif Take elements from an array along an axis.
 *
 *        Slice along a particular axis with the provided indices. E.g., given an input
 *        with shape ``(d0, d1, d2)`` and indices with shape ``(i0, i1)``, then the output
 *        will have shape ``(i0, i1, d1, d2)``, with::
 *
 *        output[i,j,:,:] = input[indices[i,j],:,:]
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *
 *        [[ 3.,  4.],
 *        [ 5.,  6.]]]
 *
 *        .. note::
 *        Only slicing axis 0 is supported now.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/indexing_op.cc:L79
 * \param symbol_name name of the resulting symbol
 * \param a The source array.
 * \param indices The indices of the values to extract.
 * \param axis the axis of data tensor to be taken.
 * \param mode specify how out-of-bound indices bahave.
 * \return new symbol
 */
inline Symbol take(const std::string& symbol_name,
                   Symbol a,
                   Symbol indices,
                   int axis = 0,
                   TakeMode mode = TakeMode::raise) {
  static const char *TakeModeValues[] = {
    "clip",
    "raise",
    "wrap"
  };
  return Operator("take")
           .SetParam("axis", axis)
           .SetParam("mode", TakeModeValues[int(mode)])
           .SetInput("a", a)
           .SetInput("indices", indices)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Take elements from a data batch.
 *
 *        Given an ``(d0, d1)`` input array, and ``(d0,)`` indices, the output will be a
 *        ``(d0,)`` computed by::
 *
 *        output[i] = input[i, indices[i]]
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        batch_take(x, [0,1,0]) = [ 1.  4.  5.]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/indexing_op.cc:L131
 * \param symbol_name name of the resulting symbol
 * \param a Input data array
 * \param indices index array
 * \return new symbol
 */
inline Symbol batch_take(const std::string& symbol_name,
                         Symbol a,
                         Symbol indices) {
  return Operator("batch_take")
           .SetInput("a", a)
           .SetInput("indices", indices)
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output
 */
enum class One_hotDtype {
  float16 = 0,
  float32 = 1,
  float64 = 2,
  int32 = 3,
  uint8 = 4
};

/*!
 * \breif Returns a one-hot array.
 *
 *        The locations represented by ``indices`` take value ``on_value``, while all
 *        other locations take value ``off_value``.
 *
 *        Assume ``indices`` has shape ``(i0, i1)``, then the output will have shape
 *        ``(i0, i1, depth)`` and::
 *
 *        output[i,j,:] = off_value
 *        output[i,j,indices[i,j]] = on_value
 *
 *        Examples::
 *
 *        one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]
 *        [ 1.  0.  0.]
 *        [ 0.  0.  1.]
 *        [ 1.  0.  0.]]
 *
 *        one_hot([1,0,2,0], 3, on_value=8, off_value=1,
 *        dtype='int32') = [[1 8 1]
 *        [8 1 1]
 *        [1 1 8]
 *        [8 1 1]]
 *
 *        one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]
 *        [ 1.  0.  0.]]
 *
 *        [[ 0.  1.  0.]
 *        [ 1.  0.  0.]]
 *
 *        [[ 0.  0.  1.]
 *        [ 1.  0.  0.]]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/indexing_op.cc:L177
 * \param symbol_name name of the resulting symbol
 * \param indices array of locations where to set on_value
 * \param depth The dimension size at dim = axis.
 * \param on_value The value assigned to the locations represented by indices.
 * \param off_value The value assigned to the locations not represented by indices.
 * \param dtype DType of the output
 * \return new symbol
 */
inline Symbol one_hot(const std::string& symbol_name,
                      Symbol indices,
                      int depth,
                      double on_value = 1,
                      double off_value = 0,
                      One_hotDtype dtype = One_hotDtype::float32) {
  static const char *One_hotDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("one_hot")
           .SetParam("depth", depth)
           .SetParam("on_value", on_value)
           .SetParam("off_value", off_value)
           .SetParam("dtype", One_hotDtypeValues[int(dtype)])
           .SetInput("indices", indices)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Reshape array into a new shape.
 *
 *        The shape is a tuple of int such as (2,3,4). The new shape should not change the
 *        array size. For example::
 *
 *        reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]
 *
 *        In addition, we can use special codes, which are integers less than
 *        1, on some shape dimensions. To inference the output shape, we set it to an
 *        empty tuple at beginning. When continuously pop dimensions from the original
 *        shape starting from the beginning, and then push translated results into the
 *        shape.
 *
 *        Each special code presents a way of translation.
 *
 *        - ``0`` for copying one. Pop one input dimension and push into the output. For
 *
 *        - input=(2,3,4), shape=(4,0,2), output=(4,3,2)
 *        - input=(2,3,4), shape=(2,0,0), output=(2,3,4)
 *
 *        - ``-1`` for inference. Push a placeholder into the output whose value will be
 *
 *        - input=(2,3,4), shape=(6,1,-1), output=(6,1,4)
 *        - input=(2,3,4), shape=(3,-1,8), output=(3,1,8)
 *        - input=(2,3,4), shape=(-1,), output=(24,)
 *
 *        - ``-2`` for copying all. Pop all remaining input dimensions and push them into
 *        the output::
 *
 *        - input=(2,3,4), shape=(-2), output=(9,8,7)
 *        - input=(2,3,4), shape=(2,-2), output=(2,3,4)
 *        - input=(2,3,4), shape=(-2,1,1), output=(2,3,4,1,1)
 *
 *        - ``-3`` for merging two dimensions. Pop two input dimensions, compute the
 *        push into the output::
 *
 *        - input=(2,3,4), shape=(-3,4), output=(6,4)
 *        - input=(2,3,4), shape=(0,-3), output=(2,12)
 *        - input=(2,3,4), shape=(-3,-2), output=(6,4)
 *
 *        - ``-4`` for splitting two dimensions. Pop one input dimensions, next split it
 *        according to the next two dimensions (can contain one ``-1``) specified after
 *        this code, then push into the output::
 *
 *        - input=(2,3,4), shape=(-4,1,2,-2), output=(1,2,3,4)
 *        - input=(2,3,4), shape=(2,-4,-1,3,-2), output=(2,1,3,4)
 *
 *        If the argument ``reverse`` is set to be true, then translating the input shape
 *        from right to left. For example, with input shape (10, 5, 4) target shape (-1,
 *        0), then the output shape will be (50,4) if ``reverse=1``, otherwise it will be
 *        (40,5).
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L78
 * \param symbol_name name of the resulting symbol
 * \param data Input data to reshape.
 * \param target_shape (Deprecated! Use ``shape`` instead.) Target new shape. One and
 * \param keep_highest (Deprecated! Use ``shape`` instead.) Whether keep the highest dim
 *        unchanged.If set to true, then the first dim in target_shape is ignored,and
 * \param shape The target shape
 * \param reverse If true then translating the input shape from right to left
 * \return new symbol
 */
inline Symbol Reshape(const std::string& symbol_name,
                      Symbol data,
                      Shape target_shape = Shape(0,0),
                      bool keep_highest = false,
                      Shape shape = Shape(),
                      bool reverse = false) {
  return Operator("Reshape")
           .SetParam("target_shape", target_shape)
           .SetParam("keep_highest", keep_highest)
           .SetParam("shape", shape)
           .SetParam("reverse", reverse)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Flatten input into a 2-D array by collapsing the higher dimensions.
 *
 *        Assume the input array has shape ``(d1, d2, ..., dk)``, then ``flatten``
 *        the input array into shape ``(d1, d2*...*dk)``.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L101
 * \param symbol_name name of the resulting symbol
 * \param data Input data to reshape.
 * \return new symbol
 */
inline Symbol Flatten(const std::string& symbol_name,
                      Symbol data) {
  return Operator("Flatten")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Permute the dimensions of an array.
 *
 *        Examples::
 *
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *
 *        transpose(x) = [[ 1.,  3.],
 *        [ 2.,  4.]]
 *
 *        x = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *
 *        [[ 5.,  6.],
 *        [ 7.,  8.]]]
 *
 *        transpose(x) = [[[ 1.,  5.],
 *        [ 3.,  7.]],
 *
 *        [[ 2.,  6.],
 *        [ 4.,  8.]]]
 *
 *        transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
 *        [ 5.,  6.]],
 *
 *        [[ 3.,  4.],
 *        [ 7.,  8.]]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L142
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axes Target axis order. By default the axes will be inverted.
 * \return new symbol
 */
inline Symbol transpose(const std::string& symbol_name,
                        Symbol data,
                        Shape axes = Shape()) {
  return Operator("transpose")
           .SetParam("axes", axes)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Insert a new axis with size 1 into the array shape
 *
 *        For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``
 *        will return a new array with shape ``(2,1,3,4)``.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L175
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Position (amongst axes) where new axis is to be inserted.
 * \return new symbol
 */
inline Symbol expand_dims(const std::string& symbol_name,
                          Symbol data,
                          uint32_t axis) {
  return Operator("expand_dims")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Crop a continuous region from the array.
 *
 *        Assume the input array has *n* dimensions, given ``begin=(b_1, ..., b_n)`` and
 *        ``end=(e_1, ..., e_n)``, then ``crop`` will return a region with shape
 *        ``(e_1-b_1, ..., e_n-b_n)``. The result's *k*-th dimension contains elements
 *        from the *k*-th dimension of the input array with the open range ``[b_k, e_k)``.
 *
 *        For example::
 *
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        crop(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
 *        [ 6.,  7.,  8.]]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L207
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param begin starting coordinates
 * \param end ending coordinates
 * \return new symbol
 */
inline Symbol slice(const std::string& symbol_name,
                    Symbol data,
                    Shape begin,
                    Shape end) {
  return Operator("slice")
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Slice along a given axis.
 *
 *        Examples:
 *
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],
 *        [  5.,   6.],
 *        [  9.,  10.]]
 *
 *        slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],
 *        [  6.,   7.],
 *        [ 10.,  11.]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L285
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis The axis to be sliced. Negative axis means to count from the last to the
 * \param begin The beginning index to be sliced. Negative values are interpreted as
 * \param end The end index to be sliced. The end can be None, in which case all the rest
 *        elements are used. Also, negative values are interpreted as counting from the
 * \return new symbol
 */
inline Symbol slice_axis(const std::string& symbol_name,
                         Symbol data,
                         int axis,
                         int begin,
                         dmlc::optional<int> end) {
  return Operator("slice_axis")
           .SetParam("axis", axis)
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Dot product of two arrays.
 *
 *        ``dot``'s behavior depends on the input array dimensions:
 *
 *        - 1-D arrays: inner product of vectors
 *        - 2-D arrays: matrix multiplication
 *        - N-D arrays: a sum product over the last axis of the first input and the first
 *        axis of the second input
 *
 *        For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape
 *        result array will have shape `(n,m,r,s)`. It is computed by::
 *
 *        dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L318
 * \param symbol_name name of the resulting symbol
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \return new symbol
 */
inline Symbol dot(const std::string& symbol_name,
                  Symbol lhs,
                  Symbol rhs,
                  bool transpose_a = false,
                  bool transpose_b = false) {
  return Operator("dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Batchwise dot product.
 *
 *        ``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
 *        ``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.
 *
 *        For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape
 *        `(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,
 *        which is computed by::
 *
 *        batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L354
 * \param symbol_name name of the resulting symbol
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \return new symbol
 */
inline Symbol batch_dot(const std::string& symbol_name,
                        Symbol lhs,
                        Symbol rhs,
                        bool transpose_a = false,
                        bool transpose_b = false) {
  return Operator("batch_dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Clip (limit) the values in an array, elementwise
 *
 *        Given an interval, values outside the interval are clipped to the interval
 *        edges. That is::
 *
 *        clip(x) = max(min(x, a_max)), a_min)
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L393
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param a_min Minimum value
 * \param a_max Maximum value
 * \return new symbol
 */
inline Symbol clip(const std::string& symbol_name,
                   Symbol data,
                   mx_float a_min,
                   mx_float a_max) {
  return Operator("clip")
           .SetParam("a_min", a_min)
           .SetParam("a_max", a_max)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Repeat elements of an array.
 *
 *        In default, ``repeat`` flatten the input array into 1-D and then repeat the
 *        elements::
 *
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *
 *        repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]
 *
 *        We can also choose a particular axis to repeat, in which a negative axis is
 *        interpreted counting from the backward::
 *
 *        repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *
 *        repeat(x, repeats=2, axis=-1) = [[ 1.,  2.],
 *        [ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 3.,  4.]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L432
 * \param symbol_name name of the resulting symbol
 * \param data Input data array
 * \param repeats The number of repetitions for each element.
 * \param axis The axis along which to repeat values. The negative numbers are
 *        interpreted counting from the backward. By default, use the flattened input
 * \return new symbol
 */
inline Symbol repeat(const std::string& symbol_name,
                     Symbol data,
                     int repeats,
                     dmlc::optional<int> axis = dmlc::optional<int>()) {
  return Operator("repeat")
           .SetParam("repeats", repeats)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Repeat the whole array by multiple times.
 *
 *        If ``reps`` has length *d*, and input array has dimension of *n*. There are
 *        there cases:
 *
 *        - **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::
 *
 *        x = [[1, 2],
 *        [3, 4]]
 *
 *        tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]
 *
 *        - **n>d**. ``reps`` is promoted to length *n* by pre-pending 1’s to it. Thus
 *        an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::
 *
 *
 *        tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.]]
 *
 *        - **n<d**. The input is promoted to be d-dimensional by prepending new axes. So
 *        shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::
 *
 *        tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]],
 *
 *        [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L489
 * \param symbol_name name of the resulting symbol
 * \param data Input data array
 * \param reps The number of times for repeating the tensor a. If reps has length d, the
 *        result will have dimension of max(d, a.ndim); If a.ndim < d, a is promoted to
 *        be d-dimensional by prepending new axes. If a.ndim > d, reps is promoted to
 * \return new symbol
 */
inline Symbol tile(const std::string& symbol_name,
                   Symbol data,
                   Shape reps) {
  return Operator("tile")
           .SetParam("reps", reps)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Reverse elements of an array with axis
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:512
 * \param symbol_name name of the resulting symbol
 * \param data Input data array
 * \param axis The axis which to reverse elements.
 * \return new symbol
 */
inline Symbol reverse(const std::string& symbol_name,
                      Symbol data,
                      Shape axis) {
  return Operator("reverse")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output. If output given, set to type of output.If output not given
 */
enum class UniformDtype {
  None = 0,
  float16 = 1,
  float32 = 2,
  float64 = 3
};

/*!
 * \breif Draw samples from a uniform distribution.
 *
 *        Samples are uniformly distributed over the half-open interval [low, high)
 *        (includes low, but excludes high)::
 *
 *        nd.uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
 *        [ 0.54488319,  0.84725171]]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/sample_op.cc:L24
 * \param symbol_name name of the resulting symbol
 * \param low The lower bound of distribution
 * \param high The upper bound of distribution
 * \param shape The shape of the output
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for
 * \param dtype DType of the output. If output given, set to type of output.If output not
 * \return new symbol
 */
inline Symbol uniform(const std::string& symbol_name,
                      mx_float low = 0,
                      mx_float high = 1,
                      Shape shape = Shape(),
                      const std::string& ctx = "",
                      UniformDtype dtype = UniformDtype::None) {
  static const char *UniformDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("uniform")
           .SetParam("low", low)
           .SetParam("high", high)
           .SetParam("shape", shape)
           .SetParam("dtype", UniformDtypeValues[int(dtype)])
           .CreateSymbol(symbol_name);
}

/*! \breif DType of the output. If output given, set to type of output.If output not given
 */
enum class NormalDtype {
  None = 0,
  float16 = 1,
  float32 = 2,
  float64 = 3
};

/*!
 * \breif Draw random samples from a normal (Gaussian) distribution.
 *
 *        Examples::
 *
 *        normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
 *        [-1.23474145,  1.55807114]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/sample_op.cc:L35
 * \param symbol_name name of the resulting symbol
 * \param loc Mean of the distribution.
 * \param scale Standard deviation of the distribution.
 * \param shape The shape of the output
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for
 * \param dtype DType of the output. If output given, set to type of output.If output not
 * \return new symbol
 */
inline Symbol normal(const std::string& symbol_name,
                     mx_float loc = 0,
                     mx_float scale = 1,
                     Shape shape = Shape(),
                     const std::string& ctx = "",
                     NormalDtype dtype = NormalDtype::None) {
  static const char *NormalDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("normal")
           .SetParam("loc", loc)
           .SetParam("scale", scale)
           .SetParam("shape", shape)
           .SetParam("dtype", NormalDtypeValues[int(dtype)])
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the indices of the maximum values along an axis.
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/broadcast_reduce_op_index.cc:11
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis Empty or unsigned. The axis to perform the reduction.If left empty, a
 * \param keepdims If true, the axis which is reduced is left in the result as dimension
 * \return new symbol
 */
inline Symbol argmax(const std::string& symbol_name,
                     Symbol data,
                     int axis = -1,
                     bool keepdims = false) {
  return Operator("argmax")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the indices of the minimum values along an axis.
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/broadcast_reduce_op_index.cc:16
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis Empty or unsigned. The axis to perform the reduction.If left empty, a
 * \param keepdims If true, the axis which is reduced is left in the result as dimension
 * \return new symbol
 */
inline Symbol argmin(const std::string& symbol_name,
                     Symbol data,
                     int axis = -1,
                     bool keepdims = false) {
  return Operator("argmin")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 * \param symbol_name name of the resulting symbol
 * \param src Source input
 * \return new symbol
 */
inline Symbol argmax_channel(const std::string& symbol_name,
                             Symbol src) {
  return Operator("argmax_channel")
           .SetInput("src", src)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute the sum of array elements over given axes.
 *
 *        The argument ``axis`` specifies the axes to compute over:
 *
 *        - **()**: compute over all elements into a scalar array with shape ``(1,)``.
 *        the default option.
 *        - **int**: compute over along a particular axis. If input has shape ``(n, m,
 *        use ``axis=0`` will result in an array with shape ``(m, k)``.
 *        - **tuple of int**: compute over multiple axes. Again assume input shape ``(n,
 *        k)``, with ``axis=(0,2)`` we obtain a ``(m,)`` shape array.
 *
 *        If ``keepdims = 1``, then the result array will has the same number of
 *        as the input, while the reduced axes will have size 1.
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol sum(const std::string& symbol_name,
                  Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false) {
  return Operator("sum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute the mean of array elements over given axes.
 *
 *        The argument ``axis`` specifies the axes to compute over:
 *
 *        - **()**: compute over all elements into a scalar array with shape ``(1,)``.
 *        the default option.
 *        - **int**: compute over along a particular axis. If input has shape ``(n, m,
 *        use ``axis=0`` will result in an array with shape ``(m, k)``.
 *        - **tuple of int**: compute over multiple axes. Again assume input shape ``(n,
 *        k)``, with ``axis=(0,2)`` we obtain a ``(m,)`` shape array.
 *
 *        If ``keepdims = 1``, then the result array will has the same number of
 *        as the input, while the reduced axes will have size 1.
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol mean(const std::string& symbol_name,
                   Symbol data,
                   Shape axis = Shape(),
                   bool keepdims = false) {
  return Operator("mean")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute the product of array elements over given axes.
 *
 *        The argument ``axis`` specifies the axes to compute over:
 *
 *        - **()**: compute over all elements into a scalar array with shape ``(1,)``.
 *        the default option.
 *        - **int**: compute over along a particular axis. If input has shape ``(n, m,
 *        use ``axis=0`` will result in an array with shape ``(m, k)``.
 *        - **tuple of int**: compute over multiple axes. Again assume input shape ``(n,
 *        k)``, with ``axis=(0,2)`` we obtain a ``(m,)`` shape array.
 *
 *        If ``keepdims = 1``, then the result array will has the same number of
 *        as the input, while the reduced axes will have size 1.
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol prod(const std::string& symbol_name,
                   Symbol data,
                   Shape axis = Shape(),
                   bool keepdims = false) {
  return Operator("prod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute the sum of array elements over given axes with ``NaN`` ignored
 *
 *        Refer to ``sum`` for more details.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol nansum(const std::string& symbol_name,
                     Symbol data,
                     Shape axis = Shape(),
                     bool keepdims = false) {
  return Operator("nansum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute the product of array elements over given axes with ``NaN`` ignored
 *
 *        Refer to ``prod`` for more details.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol nanprod(const std::string& symbol_name,
                      Symbol data,
                      Shape axis = Shape(),
                      bool keepdims = false) {
  return Operator("nanprod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute the max of array elements over given axes.
 *
 *        The argument ``axis`` specifies the axes to compute over:
 *
 *        - **()**: compute over all elements into a scalar array with shape ``(1,)``.
 *        the default option.
 *        - **int**: compute over along a particular axis. If input has shape ``(n, m,
 *        use ``axis=0`` will result in an array with shape ``(m, k)``.
 *        - **tuple of int**: compute over multiple axes. Again assume input shape ``(n,
 *        k)``, with ``axis=(0,2)`` we obtain a ``(m,)`` shape array.
 *
 *        If ``keepdims = 1``, then the result array will has the same number of
 *        as the input, while the reduced axes will have size 1.
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol max(const std::string& symbol_name,
                  Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false) {
  return Operator("max")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute the min of array elements over given axes.
 *
 *        The argument ``axis`` specifies the axes to compute over:
 *
 *        - **()**: compute over all elements into a scalar array with shape ``(1,)``.
 *        the default option.
 *        - **int**: compute over along a particular axis. If input has shape ``(n, m,
 *        use ``axis=0`` will result in an array with shape ``(m, k)``.
 *        - **tuple of int**: compute over multiple axes. Again assume input shape ``(n,
 *        k)``, with ``axis=(0,2)`` we obtain a ``(m,)`` shape array.
 *
 *        If ``keepdims = 1``, then the result array will has the same number of
 *        as the input, while the reduced axes will have size 1.
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol min(const std::string& symbol_name,
                  Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false) {
  return Operator("min")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Broadcast an array over particular axes.
 *
 *        Broadcasting is allowed on axes which size 1, such as from ``(2,1,3,1)`` to
 *        ``(2,8,3,9)``. Elemenets will be duplicated on the broadcasted axes.
 *
 *        For example::
 *
 *        // given (1,2,1) shape x
 *        x = [[[ 1.],
 *        [ 2.]]]
 *
 *        // broadcast on axis 2
 *        broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]]
 *        // broadcast on axes 0 and 2
 *        broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]],
 *        [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]]
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axes to perform the broadcasting.
 * \param size Target sizes of the broadcasting axes.
 * \return new symbol
 */
inline Symbol broadcast_axis(const std::string& symbol_name,
                             Symbol data,
                             Shape axis = Shape(),
                             Shape size = Shape()) {
  return Operator("broadcast_axis")
           .SetParam("axis", axis)
           .SetParam("size", size)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Broadcast an array to a new shape.
 *
 *        Broadcasting is allowed on axes which size 1, such as from ``(2,1,3,1)`` to
 *        ``(2,8,3,9)``. Elemenets will be duplicated on the broadcasted axes.
 *
 *        For example::
 *
 *        broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
 *        [ 1.,  2.,  3.]])
 *
 *        The dimensions that will not be changed can also use the special code ``0`` that
 *        means copy the original value. So with ``shape=(2,0)`` we will obtain the same
 *        results in the above example.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param shape The shape of the desired array. We can set the dim to zero if it's same
 *        as the original. E.g `A = broadcast_to(B, shape=(10, 0, 0))` has the same
 * \return new symbol
 */
inline Symbol broadcast_to(const std::string& symbol_name,
                           Symbol data,
                           Shape shape = Shape()) {
  return Operator("broadcast_to")
           .SetParam("shape", shape)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Compute the L2 norm.
 *
 *        Flatten then input array and then compute the l2 norm.
 *
 *        Examples::
 *
 *        x = [[1, 2],
 *        [3, 4]]
 *
 *        norm(x) = [5.47722578]
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param src Source input
 * \return new symbol
 */
inline Symbol norm(const std::string& symbol_name,
                   Symbol src) {
  return Operator("norm")
           .SetInput("src", src)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Given three ndarrays, condition, x, and y, return an ndarray with the elements
 *        from x or y, depending on the elements from condition are true or false. x and
 *        y must have the same shape. If condition has the same shape as x, each element
 *        in the output array is from x if the corresponding element in the condition is
 *        true, and from y if false. If condtion does not have the same shape as x, it
 *        must be a 1D array whose size is the same as x's first dimension size. Each row
 *        of the output array is from x's row if the corresponding element from condition
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/control_flow_op.cc:21
 * \param symbol_name name of the resulting symbol
 * \param condition condition array
 * \param x
 * \param y
 * \return new symbol
 */
inline Symbol where(const std::string& symbol_name,
                    Symbol condition,
                    Symbol x,
                    Symbol y) {
  return Operator("where")
           .SetInput("condition", condition)
           .SetInput("x", x)
           .SetInput("y", y)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Add arguments, element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_add(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Substract arguments, element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_sub(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_sub")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Multiply arguments, element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_mul(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Divide arguments, element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_div(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif First array elements raised to powers from second array,
 *        element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 *        /home/xlidc/mxnet/src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L16
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_power(const std::string& symbol_name,
                              Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_power")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Element-wise maximum of array elements with broadcasting.
 *
 *
 *
 *        Defined in
 *        /home/xlidc/mxnet/src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L34
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_maximum(const std::string& symbol_name,
                                Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_maximum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Element-wise minimum of array elements with broadcasting.
 *
 *
 *
 *        Defined in
 *        /home/xlidc/mxnet/src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L52
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_minimum(const std::string& symbol_name,
                                Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_minimum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Given the "legs" of a right triangle, return its hypotenuse
 *        with broadcasting.
 *
 *
 *
 *        Defined in
 *        /home/xlidc/mxnet/src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L71
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_hypot(const std::string& symbol_name,
                              Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_hypot")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Return (lhs == rhs), element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_equal(const std::string& symbol_name,
                              Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Return (lhs != rhs), element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_not_equal(const std::string& symbol_name,
                                  Symbol lhs,
                                  Symbol rhs) {
  return Operator("broadcast_not_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Return (lhs > rhs), element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_greater(const std::string& symbol_name,
                                Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_greater")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Return (lhs >= rhs), element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_greater_equal(const std::string& symbol_name,
                                      Symbol lhs,
                                      Symbol rhs) {
  return Operator("broadcast_greater_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Return (lhs < rhs), element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_lesser(const std::string& symbol_name,
                               Symbol lhs,
                               Symbol rhs) {
  return Operator("broadcast_lesser")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Return (lhs <= rhs), element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_lesser_equal(const std::string& symbol_name,
                                     Symbol lhs,
                                     Symbol rhs) {
  return Operator("broadcast_lesser_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Add all input arguments element-wise.
 *
 *        .. math::
 *        add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n
 *
 *        ``add_n`` is potentially more efficient than calling ``add`` by `n` times.
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_sum.cc:L63
 * \param symbol_name name of the resulting symbol
 * \param args Positional input arguments
 * \return new symbol
 */
inline Symbol add_n(const std::string& symbol_name,
                    const std::vector<Symbol>& args) {
  return Operator("add_n")
(args)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Custom operator implemented in frontend.
 * \param symbol_name name of the resulting symbol
 * \param op_type Type of custom operator. Must be registered first.
 * \return new symbol
 */
inline Symbol Custom(const std::string& symbol_name,
                     const std::string& op_type) {
  return Operator("Custom")
           .CreateSymbol(symbol_name);
}

/*! \breif Activation function to be applied.
 */
enum class ActivationActType {
  relu = 0,
  sigmoid = 1,
  softrelu = 2,
  tanh = 3
};

/*!
 * \breif Elementwise activation function.
 *        The activation operations are applied elementwisely to each array elements. The
 *        following types are supported:
 *
 *        - `relu`: Rectified Linear Unit, `y = max(x, 0)`
 *        - `sigmoid`: `y = 1 / (1 + exp(-x))`
 *        - `tanh`: Hyperbolic tangent, `y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
 *        - `softrelu`: Soft ReLU, or SoftPlus, `y = log(1 + exp(x))`
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/activation.cc:L76
 * \param symbol_name name of the resulting symbol
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \return new symbol
 */
inline Symbol Activation(const std::string& symbol_name,
                         Symbol data,
                         ActivationActType act_type) {
  static const char *ActivationActTypeValues[] = {
    "relu",
    "sigmoid",
    "softrelu",
    "tanh"
  };
  return Operator("Activation")
           .SetParam("act_type", ActivationActTypeValues[int(act_type)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply bilinear sampling to input feature map, which is the key of "[NIPS2015]
 *        output[batch, channel, y_dst, x_dst] = G(data[batch, channel, y_src, x_src)
 *        x_dst, y_dst enumerate all spatial locations in output
 *        x_src = grid[batch, 0, y_dst, x_dst]
 *        y_src = grid[batch, 1, y_dst, x_dst]
 *        G() denotes the bilinear interpolation kernel
 *        The out-boundary points will be padded as zeros. (The boundary is defined to be
 *        The shape of output will be (data.shape[0], data.shape[1], grid.shape[2],
 *        The operator assumes that grid has been nomalized. If you want to design a
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the BilinearsamplerOp.
 * \param grid Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src
 * \return new symbol
 */
inline Symbol BilinearSampler(const std::string& symbol_name,
                              Symbol data,
                              Symbol grid) {
  return Operator("BilinearSampler")
           .SetInput("data", data)
           .SetInput("grid", grid)
           .CreateSymbol(symbol_name);
}

/*! \breif Whether to pick convolution algo by running performance test.
 */
enum class ConvolutionCudnnTune {
  None = 0,
  fastest = 1,
  limited_workspace = 2,
  off = 3
};

/*! \breif Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 */
enum class ConvolutionLayout {
  None = 0,
  NCDHW = 1,
  NCHW = 2,
  NDHWC = 3,
  NHWC = 4
};

/*!
 * \breif Compute *N*-D convolution on *(N+2)*-D input.
 *
 *        In the simplest 2-D convolution, given input data with shape *(batch_size,
 *        channel, height, weight)*, the output is computed by
 *
 *        .. math::
 *
 *        out[n,i,:,:] = bias[i] + \sum_{j=0}^{num\_filter} data[n,j,:,:] \star
 *        weight[i,j,:,:]
 *
 *        where :math:`\star` is the 2-D cross-correlation operator.
 *
 *        For general 2-D convolution, the shapes are
 *
 *        - **data**: *(batch_size, channel, height, weight)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_height, out_weight)*.
 *
 *        Define::
 *
 *        f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1
 *
 *        then we have::
 *
 *        out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
 *        out_weight=f(weight, kernel[1], pad[1], stride[1], dilate[1])
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *        The default data ``layout`` is *NCHW*, namely *(batch_size, channle, height,
 *        weight)*. We can choose other layouts such as *NHWC*.
 *
 *        If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
 *        evenly into *g* parts along the channel axis, and also evenly split ``weight``
 *        along the first dimension. Next compute the convolution on the *i*-th part of
 *        the data with the *i*-th weight part. The output is obtained by concating all
 *        the *g* results.
 *
 *        To perform 1-D convolution, simply use 2-D convolution but set the last axis
 *        size to be 1 for both data and weight.
 *
 *        3-D convolution adds an additional depth dimension besides height and
 *        weight. The shapes are
 *
 *        - **data**: *(batch_size, channel, depth, height, weight)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_depth, out_height, out_weight)*.
 *
 *        Both ``weight`` and ``bias`` are learnable parameters.
 *
 *        There are other options to tune the performance.
 *
 *        - **cudnn_tune**: enable this option leads to higher startup time but may give
 *        faster speed. Options are
 *
 *        - **off**: no tuning
 *        - **limited_workspace**:run test and pick the fastest algorithm that doesn't
 *        exceed workspace limit.
 *        - **fastest**: pick the fastest algorithm and ignore workspace limit.
 *        - **None** (default): the behavior is determined by environment variable
 *        ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
 *        (default), 2 for fastest.
 *
 *        - **workspace**: A large number leads to more (GPU) memory usage but may improve
 *        the performance.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/convolution.cc:L143
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the ConvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions.
 * \param workspace Maximum temperal workspace allowed for convolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution(const std::string& symbol_name,
                          Symbol data,
                          Symbol weight,
                          Symbol bias,
                          Shape kernel,
                          uint32_t num_filter,
                          Shape stride = Shape(),
                          Shape dilate = Shape(),
                          Shape pad = Shape(),
                          uint32_t num_group = 1,
                          uint64_t workspace = 1024,
                          bool no_bias = false,
                          ConvolutionCudnnTune cudnn_tune = ConvolutionCudnnTune::None,
                          bool cudnn_off = false,
                          ConvolutionLayout layout = ConvolutionLayout::None) {
  static const char *ConvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *ConvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", ConvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", ConvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply correlation to inputs
 * \param symbol_name name of the resulting symbol
 * \param data1 Input data1 to the correlation.
 * \param data2 Input data2 to the correlation.
 * \param kernel_size kernel size for Correlation must be an odd number
 * \param max_displacement Max displacement of Correlation
 * \param stride1 stride1 quantize data1 globally
 * \param stride2 stride2 quantize data2 within the neighborhood centered around data1
 * \param pad_size pad for Correlation
 * \param is_multiply operation type is either multiplication or subduction
 * \return new symbol
 */
inline Symbol Correlation(const std::string& symbol_name,
                          Symbol data1,
                          Symbol data2,
                          uint32_t kernel_size = 1,
                          uint32_t max_displacement = 1,
                          uint32_t stride1 = 1,
                          uint32_t stride2 = 1,
                          uint32_t pad_size = 0,
                          bool is_multiply = true) {
  return Operator("Correlation")
           .SetParam("kernel_size", kernel_size)
           .SetParam("max_displacement", max_displacement)
           .SetParam("stride1", stride1)
           .SetParam("stride2", stride2)
           .SetParam("pad_size", pad_size)
           .SetParam("is_multiply", is_multiply)
           .SetInput("data1", data1)
           .SetInput("data2", data2)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or
 *        with width and height of the second input symbol, i.e., with one input, we need
 *        h_w to specify the crop height and width, otherwise the second input symbol's
 * \param symbol_name name of the resulting symbol
 * \param data Tensor or List of Tensors, the second input will be used as crop_like
 * \param num_args Number of inputs for crop, if equals one, then we will use the h_wfor
 *        crop height and width, else if equals two, then we will use the heightand width
 * \param offset crop offset coordinate: (y, x)
 * \param h_w crop height and weight: (h, w)
 * \param center_crop If set to true, then it will use be the center_crop,or it will crop
 * \return new symbol
 */
inline Symbol Crop(const std::string& symbol_name,
                   const std::vector<Symbol>& data,
                   int num_args,
                   Shape offset = Shape(0,0),
                   Shape h_w = Shape(0,0),
                   bool center_crop = false) {
  return Operator("Crop")
           .SetParam("num_args", num_args)
           .SetParam("offset", offset)
           .SetParam("h_w", h_w)
           .SetParam("center_crop", center_crop)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply deconvolution to input then add a bias.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the DeconvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel deconvolution kernel size: (y, x)
 * \param num_filter deconvolution filter(channel) number
 * \param stride deconvolution stride: (y, x)
 * \param pad pad for deconvolution: (y, x), a good number is : (kernel-1)/2, if
 * \param adj adjustment for output shape: (y, x), if target_shape set, adj will be
 * \param target_shape output shape with targe shape : (y, x)
 * \param num_group number of groups partition
 * \param workspace Tmp workspace for deconvolution (MB)
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
inline Symbol Deconvolution(const std::string& symbol_name,
                            Symbol data,
                            Symbol weight,
                            Symbol bias,
                            Shape kernel,
                            uint32_t num_filter,
                            Shape stride = Shape(1,1),
                            Shape pad = Shape(0,0),
                            Shape adj = Shape(0,0),
                            Shape target_shape = Shape(0,0),
                            uint32_t num_group = 1,
                            uint64_t workspace = 512,
                            bool no_bias = true) {
  return Operator("Deconvolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetParam("adj", adj)
           .SetParam("target_shape", target_shape)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply dropout to input.
 *        During training, each element of the input is randomly set to zero with
 *        And then the whole tensor is rescaled by 1/(1-p) to keep the expectation the
 *        before applying dropout. During the test time, this behaves as an identity map.
 *
 * \param symbol_name name of the resulting symbol
 * \param data Input data to dropout.
 * \param p Fraction of the input that gets dropped out at training time
 * \return new symbol
 */
inline Symbol Dropout(const std::string& symbol_name,
                      Symbol data,
                      mx_float p = 0.5) {
  return Operator("Dropout")
           .SetParam("p", p)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply a linear transformation: :math:`Y = XW^T + b`.
 *
 *        Shapes:
 *
 *        - **data**: `(batch_size, input_dim)`
 *        - **weight**: `(num_hidden, input_dim)`
 *        - **bias**: `(num_hidden,)`
 *        - **out**: `(batch_size, num_hidden)`
 *
 *        The learnable parameters include both ``weight`` and ``bias``.
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/fully_connected.cc:L94
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param num_hidden Number of hidden nodes of the output.
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
inline Symbol FullyConnected(const std::string& symbol_name,
                             Symbol data,
                             Symbol weight,
                             Symbol bias,
                             int num_hidden,
                             bool no_bias = false) {
  return Operator("FullyConnected")
           .SetParam("num_hidden", num_hidden)
           .SetParam("no_bias", no_bias)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif An operator taking in a n-dimensional input tensor (n > 2), and normalizing the
 *        input by subtracting the mean and variance calculated over the spatial
 *        dimensions. This is an implemention of the operator described in "Instance
 *        Normalization: The Missing Ingredient for Fast Stylization", D. Ulyanov, A.
 *        Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2). This layer is similar to
 *        batch normalization, with two differences: first, the normalization is carried
 *        out per example ('instance'), not over a batch. Second, the same normalization
 *        is applied both at test and train time. This operation is also known as
 * \param symbol_name name of the resulting symbol
 * \param data A n-dimensional tensor (n > 2) of the form [batch, channel, spatial_dim1,
 * \param gamma A vector of length 'channel', which multiplies the normalized input.
 * \param beta A vector of length 'channel', which is added to the product of the
 * \param eps Epsilon to prevent division by 0.
 * \return new symbol
 */
inline Symbol InstanceNorm(const std::string& symbol_name,
                           Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.001) {
  return Operator("InstanceNorm")
           .SetParam("eps", eps)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol(symbol_name);
}

/*! \breif Normalization Mode. If set to instance, this operator will compute a norm for
 *        each instance in the batch; this is the default mode. If set to channel, this
 *        operator will compute a cross channel norm at each position of each instance.
 */
enum class L2NormalizationMode {
  channel = 0,
  instance = 1,
  spatial = 2
};

/*!
 * \breif Set the l2 norm of each instance to a constant.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the L2NormalizationOp.
 * \param eps Epsilon to prevent div 0
 * \param mode Normalization Mode. If set to instance, this operator will compute a norm
 *        for each instance in the batch; this is the default mode. If set to channel,
 *        this operator will compute a cross channel norm at each position of each
 * \return new symbol
 */
inline Symbol L2Normalization(const std::string& symbol_name,
                              Symbol data,
                              mx_float eps = 1e-10,
                              L2NormalizationMode mode = L2NormalizationMode::instance) {
  static const char *L2NormalizationModeValues[] = {
    "channel",
    "instance",
    "spatial"
  };
  return Operator("L2Normalization")
           .SetParam("eps", eps)
           .SetParam("mode", L2NormalizationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply convolution to input then add a bias.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the ConvolutionOp.
 * \param nsize normalization window width in elements.
 * \param alpha value of the alpha variance scaling parameter in the normalization formula
 * \param beta value of the beta power parameter in the normalization formula
 * \param knorm value of the k parameter in normalization formula
 * \return new symbol
 */
inline Symbol LRN(const std::string& symbol_name,
                  Symbol data,
                  uint32_t nsize,
                  mx_float alpha = 0.0001,
                  mx_float beta = 0.75,
                  mx_float knorm = 2) {
  return Operator("LRN")
           .SetParam("nsize", nsize)
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetParam("knorm", knorm)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif If set to null, op will not normalize on output gradient.If set to batch, op
 *        will normalize gradient by divide batch size.If set to valid, op will normalize
 */
enum class MakeLossNormalization {
  batch = 0,
  null = 1,
  valid = 2
};

/*!
 * \breif Get output from a symbol and pass 1 gradient back. This is used as a terminal
 *        loss if unary and binary operator are used to composite a loss with no
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param grad_scale gradient scale as a supplement to unary and binary operators
 * \param valid_thresh regard element valid when x > valid_thresh, this is used only in
 * \param normalization If set to null, op will not normalize on output gradient.If set
 *        to batch, op will normalize gradient by divide batch size.If set to valid, op
 * \return new symbol
 */
inline Symbol MakeLoss(const std::string& symbol_name,
                       Symbol data,
                       mx_float grad_scale = 1,
                       mx_float valid_thresh = 0,
                       MakeLossNormalization normalization = MakeLossNormalization::null) {
  static const char *MakeLossNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("MakeLoss")
           .SetParam("grad_scale", grad_scale)
           .SetParam("valid_thresh", valid_thresh)
           .SetParam("normalization", MakeLossNormalizationValues[int(normalization)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Pooling type to be applied.
 */
enum class PoolingPoolType {
  avg = 0,
  max = 1,
  sum = 2
};

/*! \breif Pooling convention to be applied.
 */
enum class PoolingPoolingConvention {
  full = 0,
  valid = 1
};

/*!
 * \breif Perform pooling on the input.
 *
 *        The shapes for 2-D pooling is
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The defintion of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor(x+2*p-k)/s+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil(x+2*p-k)/s+1
 *
 *        But ``global_pool`` is set to be true, then do a global pooling, namely reset
 *        ``kernel=(height, width)``.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *
 *        1-D pooling is special case of 2-D pooling with *weight=1* and
 *        *kernel[1]=1*.
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data will have shape *(batch_size, channel, depth,
 *        height, width)*.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/pooling.cc:L122
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param pooling_convention Pooling convention to be applied.
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling(const std::string& symbol_name,
                      Symbol data,
                      Shape kernel,
                      PoolingPoolType pool_type,
                      bool global_pool = false,
                      PoolingPoolingConvention pooling_convention = PoolingPoolingConvention::valid,
                      Shape stride = Shape(),
                      Shape pad = Shape()) {
  static const char *PoolingPoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *PoolingPoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", PoolingPoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("pooling_convention", PoolingPoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Use linear regression for final output, this is used on final output of a net.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LinearRegressionOutput(const std::string& symbol_name,
                                     Symbol data,
                                     Symbol label,
                                     mx_float grad_scale = 1) {
  return Operator("LinearRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Use mean absolute error regression for final output, this is used on final
 * \param symbol_name name of the resulting symbol
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol MAERegressionOutput(const std::string& symbol_name,
                                  Symbol data,
                                  Symbol label,
                                  mx_float grad_scale = 1) {
  return Operator("MAERegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Use Logistic regression for final output, this is used on final output of a net.
 *        Logistic regression is suitable for binary classification or probability
 * \param symbol_name name of the resulting symbol
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LogisticRegressionOutput(const std::string& symbol_name,
                                       Symbol data,
                                       Symbol label,
                                       mx_float grad_scale = 1) {
  return Operator("LogisticRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*! \breif the type of RNN to compute
 */
enum class RNNMode {
  gru = 0,
  lstm = 1,
  rnn_relu = 2,
  rnn_tanh = 3
};

/*!
 * \breif Apply a recurrent layer to input.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to RNN
 * \param parameters Vector of all RNN trainable parameters concatenated
 * \param state initial hidden state of the RNN
 * \param state_cell initial cell state for LSTM networks (only for LSTM)
 * \param state_size size of the state for each layer
 * \param num_layers number of stacked layers
 * \param mode the type of RNN to compute
 * \param bidirectional whether to use bidirectional recurrent layers
 * \param p Dropout probability, fraction of the input that gets dropped out at training
 * \param state_outputs Whether to have the states as symbol outputs.
 * \return new symbol
 */
inline Symbol RNN(const std::string& symbol_name,
                  Symbol data,
                  Symbol parameters,
                  Symbol state,
                  Symbol state_cell,
                  uint32_t state_size,
                  uint32_t num_layers,
                  RNNMode mode,
                  bool bidirectional = false,
                  mx_float p = 0,
                  bool state_outputs = false) {
  static const char *RNNModeValues[] = {
    "gru",
    "lstm",
    "rnn_relu",
    "rnn_tanh"
  };
  return Operator("RNN")
           .SetParam("state_size", state_size)
           .SetParam("num_layers", num_layers)
           .SetParam("mode", RNNModeValues[int(mode)])
           .SetParam("bidirectional", bidirectional)
           .SetParam("p", p)
           .SetParam("state_outputs", state_outputs)
           .SetInput("data", data)
           .SetInput("parameters", parameters)
           .SetInput("state", state)
           .SetInput("state_cell", state_cell)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Performs region-of-interest pooling on inputs. Resize bounding box coordinates
 *        by spatial_scale and crop input feature maps accordingly. The cropped feature
 *        maps are pooled by max pooling to a fixed size output indicated by pooled_size.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the pooling operator, a 4D Feature maps
 * \param rois Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]].
 *        (x1, y1) and (x2, y2) are top left and down right corners of designated region
 *        of interest. batch_index indicates the index of corresponding image in the
 * \param pooled_size fix pooled size: (h, w)
 * \param spatial_scale Ratio of input feature map height (or w) to raw image height (or
 * \return new symbol
 */
inline Symbol ROIPooling(const std::string& symbol_name,
                         Symbol data,
                         Symbol rois,
                         Shape pooled_size,
                         mx_float spatial_scale) {
  return Operator("ROIPooling")
           .SetParam("pooled_size", pooled_size)
           .SetParam("spatial_scale", spatial_scale)
           .SetInput("data", data)
           .SetInput("rois", rois)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Takes the last element of a sequence. Takes an n-dimensional tensor of the form
 *        [max sequence length, batchsize, other dims] and returns a (n-1)-dimensional
 *        tensor of the form [batchsize, other dims]. This operator takes an optional
 *        input tensor sequence_length of positive ints of dimension [batchsize] when the
 *        sequence_length option is set to true. This allows the operator to handle
 *        variable-length sequences. If sequence_length is false, then each example in
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input tensor of the form [max sequence length, batchsize,
 * \param sequence_length vector of sequence lengths of size batchsize
 * \param use_sequence_length If set to true, this layer takes in extra input
 * \return new symbol
 */
inline Symbol SequenceLast(const std::string& symbol_name,
                           Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false) {
  return Operator("SequenceLast")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Sets all elements outside the sequence to a constant value. Takes an
 *        n-dimensional tensor of the form [max sequence length, batchsize, other dims]
 *        and returns a tensor of the same shape. This operator takes an optional input
 *        tensor sequence_length of positive ints of dimension [batchsize] when the
 *        sequence_length option is set to true. This allows the operator to handle
 *        variable-length sequences. If sequence_length is false, then each example in
 *        the batch is assumed to have the max sequence length, and this operator becomes
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input tensor of the form [max sequence length, batchsize,
 * \param sequence_length vector of sequence lengths of size batchsize
 * \param use_sequence_length If set to true, this layer takes in extra input
 * \param value The value to be used as a mask.
 * \return new symbol
 */
inline Symbol SequenceMask(const std::string& symbol_name,
                           Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false,
                           mx_float value = 0) {
  return Operator("SequenceMask")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("value", value)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Reverses the elements of each sequence. Takes an n-dimensional tensor of the
 *        form [max sequence length, batchsize, other dims] and returns a tensor of the
 *        same shape. This operator takes an optional input tensor sequence_length of
 *        positive ints of dimension [batchsize] when the sequence_length option is set
 *        to true. This allows the operator to handle variable-length sequences. If
 *        sequence_length is false, then each example in the batch is assumed to have the
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input tensor of the form [max sequence length, batchsize,
 * \param sequence_length vector of sequence lengths of size batchsize
 * \param use_sequence_length If set to true, this layer takes in extra input
 * \return new symbol
 */
inline Symbol SequenceReverse(const std::string& symbol_name,
                              Symbol data,
                              Symbol sequence_length,
                              bool use_sequence_length = false) {
  return Operator("SequenceReverse")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*! \breif Softmax Mode. If set to instance, this operator will compute a softmax for each
 *        instance in the batch; this is the default mode. If set to channel, this
 *        operator will compute a num_channel-class softmax at each position of each
 *        instance; this can be used for fully convolutional network, image segmentation,
 */
enum class SoftmaxActivationMode {
  channel = 0,
  instance = 1
};

/*!
 * \breif Apply softmax activation to input. This is intended for internal layers. For
 *        output (loss layer) please use SoftmaxOutput. If mode=instance, this operator
 *        will compute a softmax for each instance in the batch; this is the default
 *        mode. If mode=channel, this operator will compute a num_channel-class softmax
 *        at each position of each instance; this can be used for fully convolutional
 * \param symbol_name name of the resulting symbol
 * \param data Input data to activation function.
 * \param mode Softmax Mode. If set to instance, this operator will compute a softmax for
 *        each instance in the batch; this is the default mode. If set to channel, this
 *        operator will compute a num_channel-class softmax at each position of each
 *        instance; this can be used for fully convolutional network, image segmentation,
 * \return new symbol
 */
inline Symbol SoftmaxActivation(const std::string& symbol_name,
                                Symbol data,
                                SoftmaxActivationMode mode = SoftmaxActivationMode::instance) {
  static const char *SoftmaxActivationModeValues[] = {
    "channel",
    "instance"
  };
  return Operator("SoftmaxActivation")
           .SetParam("mode", SoftmaxActivationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Normalize the gradient
 */
enum class SoftmaxOutputNormalization {
  batch = 0,
  null = 1,
  valid = 2
};

/*!
 * \breif Softmax with logit loss.
 *
 *        In the forward pass, the softmax output is returned. Assume the input data has
 *        shape *(n,k)*, then the output will have the same shape as the input, which is
 *
 *        .. math::
 *        out[i,:] = softmax(data[i,:])
 *
 *        for :math:`i=0,...,n-1`, where
 *
 *        .. math::
 *        softmax(x) = \left[..., \frac{exp(x[j])}{exp(x[0])+...+exp(x[k-1])}, ...\right]
 *
 *        For general *N*-D input array with shape :math:`(d_1, ..., d_n)`. Denoted by
 *        :math:`s=d_1d_2...d_n`. The way to compute softmax various:
 *
 *        - ``preserve_shape`` is false (default). Reshape input into a 2-D array with
 *        shape :math:`(d_1, s/d_1)` beforing computing the softmax, and then reshaped
 *        original shape.
 *
 *        - ``preserve_shape`` is true. For all :math:`i_1, ..., i_{n-1}`, compute
 *
 *        .. math::
 *        out[i_1, ..., i_{n-1}, :] = softmax(data[i_1, ..., i_{n-1},:])
 *
 *        - ``multi_output`` is true. For all :math:`i_1, ..., i_{n-1}`, compute
 *
 *        .. math::
 *        out[i_1, :, ..., i_{n-1}] = softmax(data[i_1, :, ..., i_{n-1}])
 *
 *        In the backward pass, the logit loss, also called cross-entroy loss, is
 *        added. The provided label can be a *(N-1)*-D label index array or a *N*-D label
 *        probability array.
 *
 *        Examples with a particular label can be ignored during backward by specifying
 *        ``ignore_label`` (also need ``use_ignore`` to be true).
 *
 *        A scale can be applied to the gradient by ``grad_scale``, which is often used in
 *        mutli-loss object function in which we can given each loss different weight. It
 *        also supports various ways to normalize the gradient by ``normalization``:
 *
 *        - **null**: do nothing
 *        - **batch**: divide by batch size (number of examples)
 *        - **valid**: divide by the number of examples which are not ignored.
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/softmax_output.cc:L77
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param label Ground truth label.
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the labels with value equals to ``ignore_label`` will be ignored
 * \param multi_output If set to true, softmax will applied on axis 1
 * \param use_ignore If set to true, the ignore_label value will not contribute to the
 * \param preserve_shape If true, softmax will applied on the last axis
 * \param normalization Normalize the gradient
 * \param out_grad Apply weighting from output gradient
 * \return new symbol
 */
inline Symbol SoftmaxOutput(const std::string& symbol_name,
                            Symbol data,
                            Symbol label,
                            mx_float grad_scale = 1,
                            mx_float ignore_label = -1,
                            bool multi_output = false,
                            bool use_ignore = false,
                            bool preserve_shape = false,
                            SoftmaxOutputNormalization normalization = SoftmaxOutputNormalization::null,
                            bool out_grad = false) {
  static const char *SoftmaxOutputNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("SoftmaxOutput")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxOutputNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*! \breif Normalize the gradient
 */
enum class SoftmaxNormalization {
  batch = 0,
  null = 1,
  valid = 2
};

/*!
 * \breif DEPRECATED: Perform a softmax transformation on input. Please use SoftmaxOutput
 * \param symbol_name name of the resulting symbol
 * \param data Input data to softmax.
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the labels with value equals to ``ignore_label`` will be ignored
 * \param multi_output If set to true, softmax will applied on axis 1
 * \param use_ignore If set to true, the ignore_label value will not contribute to the
 * \param preserve_shape If true, softmax will applied on the last axis
 * \param normalization Normalize the gradient
 * \param out_grad Apply weighting from output gradient
 * \return new symbol
 */
inline Symbol Softmax(const std::string& symbol_name,
                      Symbol data,
                      mx_float grad_scale = 1,
                      mx_float ignore_label = -1,
                      bool multi_output = false,
                      bool use_ignore = false,
                      bool preserve_shape = false,
                      SoftmaxNormalization normalization = SoftmaxNormalization::null,
                      bool out_grad = false) {
  static const char *SoftmaxNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("Softmax")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif transformation type
 */
enum class SpatialTransformerTransformType {
  affine = 0
};

/*! \breif sampling type
 */
enum class SpatialTransformerSamplerType {
  bilinear = 0
};

/*!
 * \breif Apply spatial transformer to input feature map.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the SpatialTransformerOp.
 * \param loc localisation net, the output dim should be 6 when transform_type is affine.
 * \param transform_type transformation type
 * \param sampler_type sampling type
 * \param target_shape output shape(h, w) of spatial transformer: (y, x)
 * \return new symbol
 */
inline Symbol SpatialTransformer(const std::string& symbol_name,
                                 Symbol data,
                                 Symbol loc,
                                 SpatialTransformerTransformType transform_type,
                                 SpatialTransformerSamplerType sampler_type,
                                 Shape target_shape = Shape(0,0)) {
  static const char *SpatialTransformerTransformTypeValues[] = {
    "affine"
  };
  static const char *SpatialTransformerSamplerTypeValues[] = {
    "bilinear"
  };
  return Operator("SpatialTransformer")
           .SetParam("transform_type", SpatialTransformerTransformTypeValues[int(transform_type)])
           .SetParam("sampler_type", SpatialTransformerSamplerTypeValues[int(sampler_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .SetInput("loc", loc)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Support Vector Machine based transformation on input, backprop L2-SVM
 * \param symbol_name name of the resulting symbol
 * \param data Input data to svm.
 * \param label Label data.
 * \param margin Scale the DType(param_.margin) for activation size
 * \param regularization_coefficient Scale the coefficient responsible for balacing
 * \param use_linear If set true, uses L1-SVM objective function. Default uses L2-SVM
 * \return new symbol
 */
inline Symbol SVMOutput(const std::string& symbol_name,
                        Symbol data,
                        Symbol label,
                        mx_float margin = 1,
                        mx_float regularization_coefficient = 1,
                        bool use_linear = false) {
  return Operator("SVMOutput")
           .SetParam("margin", margin)
           .SetParam("regularization_coefficient", regularization_coefficient)
           .SetParam("use_linear", use_linear)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*! \breif transformation type
 *        if transformation type is affine, data is affine matrix : (batch, 6)
 *        if transformation type is warp, data is optical flow : (batch, 2, h, w)
 */
enum class GridGeneratorTransformType {
  affine = 0,
  warp = 1
};

/*!
 * \breif generate sampling grid for bilinear sampling.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the GridGeneratorOp.
 * \param transform_type transformation type
 *        if transformation type is affine, data is affine matrix : (batch, 6)
 *        if transformation type is warp, data is optical flow : (batch, 2, h, w)
 * \param target_shape if transformation type is affine, the operator need a target_shape
 *        if transofrmation type is warp, the operator will ignore target_shape
 * \return new symbol
 */
inline Symbol GridGenerator(const std::string& symbol_name,
                            Symbol data,
                            GridGeneratorTransformType transform_type,
                            Shape target_shape = Shape(0,0)) {
  static const char *GridGeneratorTransformTypeValues[] = {
    "affine",
    "warp"
  };
  return Operator("GridGenerator")
           .SetParam("transform_type", GridGeneratorTransformTypeValues[int(transform_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Choose one element from each line(row for python, column for R/Julia) in lhs
 *        according to index indicated by rhs. This function assume rhs uses 0-based
 * \param symbol_name name of the resulting symbol
 * \param lhs Left operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol choose_element_0index(const std::string& symbol_name,
                                    Symbol lhs,
                                    Symbol rhs) {
  return Operator("choose_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Fill one element of each line(row for python, column for R/Julia) in lhs
 *        according to index indicated by rhs and values indicated by mhs. This function
 * \param symbol_name name of the resulting symbol
 * \param lhs Left operand to the function.
 * \param mhs Middle operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol fill_element_0index(const std::string& symbol_name,
                                  Symbol lhs,
                                  Symbol mhs,
                                  Symbol rhs) {
  return Operator("fill_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("mhs", mhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Batch normalization.
 *
 *        Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis:
 *
 *        .. math::
 *
 *        data\_mean[i] = mean(data[:,i,:,...]) \\
 *        data\_var[i] = var(data[:,i,:,...])
 *
 *        Then compute the normalized output, which has the same shape as input, as
 *
 *        .. math::
 *
 *        out[:,i,:,...] = \frac{data[:,i,:,...] -
 *
 *        Both *mean* and *var* returns a scalar by treating the input as a vector.
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        ``data_var`` as well, which are needed for the backward pass.
 *
 *        Besides the inputs and the outputs, this operator accepts two auxiliary
 *        states, ``moving_mean`` and ``moving_var``, which are *k*-length
 *        vectors. They are global statistics for the whole dataset, which are updated
 *        by::
 *
 *        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
 *        moving_var = moving_var * momentum + data_var * (1 - momentum)
 *
 *        If ``use_global_stats`` is set to be true, then ``moving_mean`` and
 *        ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
 *        the output. It is often used during inference.
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is
 *        then set ``gamma`` to 1 and its gradient to 0.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/batch_norm.cc:L84
 * \param data Input data to batch normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param eps Epsilon to prevent div 0
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of local
 * \param output_mean_var Output All,normal mean and var
 * \return new symbol
 */
inline Symbol BatchNorm(Symbol data,
                        Symbol gamma,
                        Symbol beta,
                        mx_float eps = 0.001,
                        mx_float momentum = 0.9,
                        bool fix_gamma = true,
                        bool use_global_stats = false,
                        bool output_mean_var = false) {
  return Operator("BatchNorm")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetParam("output_mean_var", output_mean_var)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol();
}

/*!
 * \breif Leaky ReLu activation
 *
 *        The following types are supported:
 *
 *        - *elu*: ``y = x > 0 ? x : slop * (exp(x)-1)``
 *        - *leaky*: ``y = x > 0 ? x : slope * x``
 *        - *prelu*: same as *leaky* but the ``slope`` is learnable.
 *        - *rrelu*: same as *leaky* but the ``slope`` is uniformly randomly chosen from
 *        *[lower_bound, upper_bound)* for training, while fixed to be
 *        *(lower_bound+upper_bound)/2* for inference.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/leaky_relu.cc:L36
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \param slope Init slope for the activation. (For leaky and elu only)
 * \param lower_bound Lower bound of random slope. (For rrelu only)
 * \param upper_bound Upper bound of random slope. (For rrelu only)
 * \return new symbol
 */
inline Symbol LeakyReLU(Symbol data,
                        LeakyReLUActType act_type = LeakyReLUActType::leaky,
                        mx_float slope = 0.25,
                        mx_float lower_bound = 0.125,
                        mx_float upper_bound = 0.334) {
  static const char *LeakyReLUActTypeValues[] = {
    "elu",
    "leaky",
    "prelu",
    "rrelu"
  };
  return Operator("LeakyReLU")
           .SetParam("act_type", LeakyReLUActTypeValues[int(act_type)])
           .SetParam("slope", slope)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Concate a list of array along a given axis.
 *
 *        The dimension sizes of the input arrays on the given axis should be the same.
 *
 *        For example::
 *
 *        x = [[1,1],[1,1]]
 *        y = [[2,2],[2,2]]
 *        z = [[3,3],[3,3],[3,3]]
 *
 *        Concat(x,y,z,dim=0) = [[ 1.,  1.],
 *        [ 1.,  1.],
 *        [ 2.,  2.],
 *        [ 2.,  2.],
 *        [ 3.,  3.],
 *        [ 3.,  3.],
 *        [ 3.,  3.]]
 *
 *        Concat(x,y,z,dim=1) = [[ 1.,  1.,  2.,  2.],
 *        [ 1.,  1.,  2.,  2.]]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/concat.cc:L69
 * \param data List of tensors to concatenate
 * \param num_args Number of inputs to be concated.
 * \param dim the dimension to be concated.
 * \return new symbol
 */
inline Symbol Concat(const std::vector<Symbol>& data,
                     int num_args,
                     int dim = 1) {
  return Operator("Concat")
           .SetParam("num_args", num_args)
           .SetParam("dim", dim)
(data)
           .CreateSymbol();
}

/*!
 * \breif Apply a sparse regularization to the output a sigmoid activation function.
 * \param data Input data.
 * \param sparseness_target The sparseness target
 * \param penalty The tradeoff parameter for the sparseness penalty
 * \param momentum The momentum for running average
 * \return new symbol
 */
inline Symbol IdentityAttachKLSparseReg(Symbol data,
                                        mx_float sparseness_target = 0.1,
                                        mx_float penalty = 0.001,
                                        mx_float momentum = 0.9) {
  return Operator("IdentityAttachKLSparseReg")
           .SetParam("sparseness_target", sparseness_target)
           .SetParam("penalty", penalty)
           .SetParam("momentum", momentum)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Calculate cross_entropy(data, one_hot(label))
 *
 *        From:/home/xlidc/mxnet/src/operator/loss_binary_op.cc:12
 * \param data Input data
 * \param label Input label
 * \return new symbol
 */
inline Symbol softmax_cross_entropy(Symbol data,
                                    Symbol label) {
  return Operator("softmax_cross_entropy")
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Pad an array.
 *
 *        Only supports 4-D and 5-D input array.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/pad.cc:L407
 * \param data An n-dimensional input tensor.
 * \param mode Padding type to use. "constant" pads all values with a constant value, the
 *        value of which can be specified with the constant_value option. "edge" uses the
 * \param pad_width A tuple of padding widths of length 2*r, where r is the rank of the
 *        input tensor, specifying number of values padded to the edges of each axis.
 *        (before_1, after_1, ... , before_N, after_N) unique pad widths for each axis.
 * \param constant_value This option is only used when mode is "constant". This value
 * \return new symbol
 */
inline Symbol Pad(Symbol data,
                  PadMode mode,
                  Shape pad_width,
                  double constant_value = 0) {
  static const char *PadModeValues[] = {
    "constant",
    "edge"
  };
  return Operator("Pad")
           .SetParam("mode", PadModeValues[int(mode)])
           .SetParam("pad_width", pad_width)
           .SetParam("constant_value", constant_value)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Updater function for sgd optimizer
 * \param lr learning_rate
 * \param wd weight decay
 * \param rescale_grad rescale gradient as grad = rescale_grad*grad.
 * \param clip_gradient If greater than 0, clip gradient to grad = max(min(grad,
 * \return new symbol
 */
inline Symbol sgd_update(mx_float lr,
                         mx_float wd = 0,
                         mx_float rescale_grad = 1,
                         mx_float clip_gradient = -1) {
  return Operator("sgd_update")
           .SetParam("lr", lr)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .CreateSymbol();
}

/*!
 * \breif Updater function for sgd optimizer
 * \param lr learning_rate
 * \param momentum momentum
 * \param wd weight decay
 * \param rescale_grad rescale gradient as grad = rescale_grad*grad.
 * \param clip_gradient If greater than 0, clip gradient to grad = max(min(grad,
 * \return new symbol
 */
inline Symbol sgd_mom_update(mx_float lr,
                             mx_float momentum = 0,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1) {
  return Operator("sgd_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .CreateSymbol();
}

/*!
 * \breif Updater function for adam optimizer
 * \param lr learning_rate
 * \param beta1 beta1
 * \param beta2 beta2
 * \param epsilon epsilon
 * \param wd weight decay
 * \param rescale_grad rescale gradient as grad = rescale_grad*grad.
 * \param clip_gradient If greater than 0, clip gradient to grad = max(min(grad,
 * \return new symbol
 */
inline Symbol adam_update(mx_float lr,
                          mx_float beta1 = 0.9,
                          mx_float beta2 = 0.999,
                          mx_float epsilon = 1e-08,
                          mx_float wd = 0,
                          mx_float rescale_grad = 1,
                          mx_float clip_gradient = -1) {
  return Operator("adam_update")
           .SetParam("lr", lr)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .CreateSymbol();
}

/*!
 * \breif Updater function for RMSProp optimizer. The RMSProp code follows the version in
 *        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 * \param lr learning_rate
 * \param gamma1 gamma1
 * \param epsilon epsilon
 * \param wd weight decay
 * \param rescale_grad rescale gradient as grad = rescale_grad*grad.
 * \param clip_gradient If greater than 0, clip gradient to grad = max(min(grad,
 * \param clip_weights If greater than 0, clip weights to weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmsprop_update(mx_float lr,
                             mx_float gamma1 = 0.95,
                             mx_float epsilon = 1e-08,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1,
                             mx_float clip_weights = -1) {
  return Operator("rmsprop_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .CreateSymbol();
}

/*!
 * \breif Updater function for RMSPropAlex optimizer. The RMSPropAlex code follows the
 *        version in http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves,
 * \param lr learning_rate
 * \param gamma1 gamma1
 * \param gamma2 gamma2
 * \param epsilon epsilon
 * \param wd weight decay
 * \param rescale_grad rescale gradient as grad = rescale_grad*grad.
 * \param clip_gradient If greater than 0, clip gradient to grad = max(min(grad,
 * \param clip_weights If greater than 0, clip weights to weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmspropalex_update(mx_float lr,
                                 mx_float gamma1 = 0.95,
                                 mx_float gamma2 = 0.9,
                                 mx_float epsilon = 1e-08,
                                 mx_float wd = 0,
                                 mx_float rescale_grad = 1,
                                 mx_float clip_gradient = -1,
                                 mx_float clip_weights = -1) {
  return Operator("rmspropalex_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("gamma2", gamma2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .CreateSymbol();
}

/*!
 * \breif Interchange two axes of an array.
 *
 *        Examples::
 *
 *        x = [[1, 2, 3]])
 *        swapaxes(x, 0, 1) = [[ 1],
 *        [ 2],
 *        [ 3]]
 *
 *        x = [[[ 0, 1],
 *        [ 2, 3]],
 *        [[ 4, 5],
 *        [ 6, 7]]]  // (2,2,2) array
 *
 *        swapaxes(x, 0, 2) = [[[ 0, 4],
 *        [ 2, 6]],
 *        [[ 1, 5],
 *        [ 3, 7]]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/swapaxis.cc:L55
 * \param data Input array.
 * \param dim1 the first axis to be swapped.
 * \param dim2 the second axis to be swapped.
 * \return new symbol
 */
inline Symbol SwapAxis(Symbol data,
                       uint32_t dim1 = 0,
                       uint32_t dim2 = 0) {
  return Operator("SwapAxis")
           .SetParam("dim1", dim1)
           .SetParam("dim2", dim2)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Split an array along a particular axis into multiple sub-arrays.
 *
 *        Assume the input array has shape ``(d_0, ..., d_n)`` and we slice it into *m*
 *        (``num_outputs=m``) subarrays along axis *k*, then we will obtain a list of *m*
 *        arrays with each of which has shape ``(d_0, ..., d_k/m, ..., d_n)``.
 *
 *        For example::
 *
 *        x = [[1, 2],
 *        [3, 4],
 *        [5, 6],
 *        [7, 8]]  // 4x2 array
 *
 *        y = split(x, axis=0, num_outputs=4) // a list of 4 arrays
 *        y[0] = [[ 1.,  2.]]  // 1x2 array
 *
 *        z = split(x, axis=0, num_outputs=2) // a list of 2 arrays
 *        z[0] = [[ 1.,  2.],
 *        [ 3.,  4.]]
 *
 *        When setting optional argument ``squeeze_axis=1``, then the *k*-dimension will
 *        be removed from the shape if it becomes 1::
 *
 *        y = split(x, axis=0, num_outputs=4, squeeze_axis=1)
 *        y[0] = [ 1.,  2.]  // (2,) vector
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/slice_channel.cc:L50
 * \param num_outputs Number of outputs to be sliced.
 * \param axis Dimension along which to slice.
 * \param squeeze_axis If true, the dimension will be squeezed. Also, input.shape[axis]
 * \return new symbol
 */
inline Symbol SliceChannel(int num_outputs,
                           int axis = 1,
                           bool squeeze_axis = false) {
  return Operator("SliceChannel")
           .SetParam("num_outputs", num_outputs)
           .SetParam("axis", axis)
           .SetParam("squeeze_axis", squeeze_axis)
           .CreateSymbol();
}

/*!
 * \breif Perform nearest neighboor/bilinear up sampling to inputs
 * \param data Array of tensors to upsample
 * \param scale Up sampling scale
 * \param sample_type upsampling method
 * \param num_args Number of inputs to be upsampled. For nearest neighbor upsampling,
 *        this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other
 *        inputs will be upsampled to thesame size. For bilinear upsampling this must be
 * \param num_filter Input filter. Only used by bilinear sample_type.
 * \param multi_input_mode How to handle multiple input. concat means concatenate
 *        upsampled images along the channel dimension. sum means add all images
 * \param workspace Tmp workspace for deconvolution (MB)
 * \return new symbol
 */
inline Symbol UpSampling(const std::vector<Symbol>& data,
                         uint32_t scale,
                         UpSamplingSampleType sample_type,
                         int num_args,
                         uint32_t num_filter = 0,
                         UpSamplingMultiInputMode multi_input_mode = UpSamplingMultiInputMode::concat,
                         uint64_t workspace = 512) {
  static const char *UpSamplingSampleTypeValues[] = {
    "bilinear",
    "nearest"
  };
  static const char *UpSamplingMultiInputModeValues[] = {
    "concat",
    "sum"
  };
  return Operator("UpSampling")
           .SetParam("scale", scale)
           .SetParam("sample_type", UpSamplingSampleTypeValues[int(sample_type)])
           .SetParam("num_args", num_args)
           .SetParam("num_filter", num_filter)
           .SetParam("multi_input_mode", UpSamplingMultiInputModeValues[int(multi_input_mode)])
           .SetParam("workspace", workspace)
(data)
           .CreateSymbol();
}

/*!
 * \breif
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_add(Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Calculate Smooth L1 Loss(lhs, scalar)
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/elemwise_binary_scalar_op_extended.cc:63
 * \param data source input
 * \param scalar scalar input
 * \return new symbol
 */
inline Symbol smooth_l1(Symbol data,
                        mx_float scalar) {
  return Operator("smooth_l1")
           .SetParam("scalar", scalar)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Return the top *k* elements in an array.
 *
 *        Examples::
 *
 *        x = [[ 0.3,  0.2,  0.4],
 *        [ 0.1,  0.3,  0.2]]
 *
 *        // return the index of the largest element on last axis
 *        topk(x) = [[ 2.],
 *        [ 1.]]
 *
 *        // return the value of the top-2 elements on last axis
 *        topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],
 *        [ 0.3,  0.2]]
 *
 *        // flatten and then return both index and value
 *        topk(x, ret_typ='both', k=2, axis=None) = [ 0.4,  0.3], [ 2.,  0.]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/ordering_op.cc:L36
 * \param src Source input
 * \param axis Axis along which to choose the top k indices. If not given, the flattened
 * \param k Number of top elements to select, should be always smaller than or equal to
 * \param ret_typ The return type. "value" means returning the top k values, "indices"
 *        means returning the indices of the top k values, "mask" means to return a mask
 *        array containing 0 and 1. 1 means the top k values. "both" means to return both
 * \param is_ascend Whether to choose k largest or k smallest. Top K largest elements
 * \return new symbol
 */
inline Symbol topk(Symbol src,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   int k = 1,
                   TopkRetTyp ret_typ = TopkRetTyp::indices,
                   bool is_ascend = false) {
  static const char *TopkRetTypValues[] = {
    "both",
    "indices",
    "mask",
    "value"
  };
  return Operator("topk")
           .SetParam("axis", axis)
           .SetParam("k", k)
           .SetParam("ret_typ", TopkRetTypValues[int(ret_typ)])
           .SetParam("is_ascend", is_ascend)
           .SetInput("src", src)
           .CreateSymbol();
}

/*!
 * \breif Return a sorted copy of an array.
 *
 *        Examples::
 *
 *        x = [[ 1, 4],
 *        [ 3, 1]]
 *
 *        // sort along the last axis
 *        sort(x) = [[ 1.,  4.],
 *        [ 1.,  3.]]
 *
 *        // flatten and then sort
 *        sort(x, axis=None) = [ 1.,  1.,  3.,  4.]
 *
 *        // sort long the first axis
 *        sort(x, axis=0) = [[ 1.,  1.],
 *        [ 3.,  4.]]
 *
 *        // in a descend order
 *        sort(x, is_ascend=0) = [[ 4.,  1.],
 *        [ 3.,  1.]]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/ordering_op.cc:L99
 * \param src Source input
 * \param axis Axis along which to choose sort the input tensor. If not given, the
 * \param is_ascend Whether sort in ascending or descending order.
 * \return new symbol
 */
inline Symbol sort(Symbol src,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   bool is_ascend = true) {
  return Operator("sort")
           .SetParam("axis", axis)
           .SetParam("is_ascend", is_ascend)
           .SetInput("src", src)
           .CreateSymbol();
}

/*!
 * \breif Returns the indices that can sort an array.
 *
 *        Examples::
 *
 *        x = [[ 0.3,  0.2,  0.4],
 *        [ 0.1,  0.3,  0.2]]
 *
 *        // sort along axis -1
 *        argsort(x) = [[ 1.,  0.,  2.],
 *        [ 0.,  2.,  1.]]
 *
 *        // sort along axis 0
 *        argsort(x, axis=0) = [[ 1.,  0.,  1.]
 *        [ 0.,  1.,  0.]]
 *
 *        // flatten and then sort
 *        argsort(x, axis=None) = [ 3.,  1.,  5.,  0.,  4.,  2.]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/ordering_op.cc:L146
 * \param src Source input
 * \param axis Axis along which to sort the input tensor. If not given, the flattened
 * \param is_ascend Whether sort in ascending or descending order.
 * \return new symbol
 */
inline Symbol argsort(Symbol src,
                      dmlc::optional<int> axis = dmlc::optional<int>(-1),
                      bool is_ascend = true) {
  return Operator("argsort")
           .SetParam("axis", axis)
           .SetParam("is_ascend", is_ascend)
           .SetInput("src", src)
           .CreateSymbol();
}

/*!
 * \breif Get output from a symbol and pass 0 gradient back
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:31
 * \param data The input
 * \return new symbol
 */
inline Symbol BlockGrad(Symbol data) {
  return Operator("BlockGrad")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Cast to a specified type, element-wise.
 *
 *        For example::
 *
 *        cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
 *        cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L65
 * \param data Source input
 * \param dtype Output data type.
 * \return new symbol
 */
inline Symbol Cast(Symbol data,
                   CastDtype dtype) {
  static const char *CastDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("Cast")
           .SetParam("dtype", CastDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Negate src
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:84
 * \param data The input
 * \return new symbol
 */
inline Symbol negative(Symbol data) {
  return Operator("negative")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the absolute value of array elements, element-wise.
 *
 *        For example:
 *        abs([-2, 0, 3]) = [2, 0, 3]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L95
 * \param data The input
 * \return new symbol
 */
inline Symbol abs(Symbol data) {
  return Operator("abs")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the indication sign of array elements, element-wise.
 *
 *        For example::
 *        sign([-2, 0, 3]) = [-1, 0, 1]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L109
 * \param data The input
 * \return new symbol
 */
inline Symbol sign(Symbol data) {
  return Operator("sign")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Round elements of the array to the nearest integer, element-wise.
 *
 *        For example::
 *        round([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -2.,  2.,  2.,  2.]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L122
 * \param data The input
 * \return new symbol
 */
inline Symbol round(Symbol data) {
  return Operator("round")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Return the ceiling of the input, element-wise.
 *
 *        For example::
 *        ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L132
 * \param data The input
 * \return new symbol
 */
inline Symbol ceil(Symbol data) {
  return Operator("ceil")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Return the floor of the input, element-wise.
 *
 *        For example::
 *        floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L141
 * \param data The input
 * \return new symbol
 */
inline Symbol floor(Symbol data) {
  return Operator("floor")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Round elements of the array to the nearest integer, element-wise.
 *
 *        For example::
 *        rint([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -2.,  1.,  2.,  2.]
 *
 *        The difference to ``round`` is that ``rint`` returns ``n`` for input ``n.5``
 *        while ``round`` returns ``n+1`` for ``n>=0``.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L154
 * \param data The input
 * \return new symbol
 */
inline Symbol rint(Symbol data) {
  return Operator("rint")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Round elements of the array to the nearest integer towards
 *        zero, element-wise.
 *
 *        For example::
 *        fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L164
 * \param data The input
 * \return new symbol
 */
inline Symbol fix(Symbol data) {
  return Operator("fix")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Calculate the square of an array, element-wise.
 *
 *        For example::
 *        square(x) = x^2
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L174
 * \param data The input
 * \return new symbol
 */
inline Symbol square(Symbol data) {
  return Operator("square")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Calculate the square-root of an array, element-wise.
 *
 *        For example::
 *        sqrt(x) = \sqrt{x}
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L187
 * \param data The input
 * \return new symbol
 */
inline Symbol sqrt(Symbol data) {
  return Operator("sqrt")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Calculate the inverse square-root of an array, element-wise.
 *
 *        For example::
 *        rsqrt(x) = 1/\sqrt{x}
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L200
 * \param data The input
 * \return new symbol
 */
inline Symbol rsqrt(Symbol data) {
  return Operator("rsqrt")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Calculate the exponential of the array, element-wise
 *
 *        For example::
 *        exp(x) = e^x \approx 2.718^x
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L215
 * \param data The input
 * \return new symbol
 */
inline Symbol exp(Symbol data) {
  return Operator("exp")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Natural logarithm, element-wise.
 *
 *        The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L225
 * \param data The input
 * \return new symbol
 */
inline Symbol log(Symbol data) {
  return Operator("log")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Calculate the base 10 logarithm of the array, element-wise.
 *
 *        ``10**log10(x) = x``
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L235
 * \param data The input
 * \return new symbol
 */
inline Symbol log10(Symbol data) {
  return Operator("log10")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Calculate the base 2 logarithm of the array, element-wise.
 *
 *        ``2**log2(x) = x``
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L245
 * \param data The input
 * \return new symbol
 */
inline Symbol log2(Symbol data) {
  return Operator("log2")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Trigonometric sine, element-wise.
 *
 *        Then input is in radians (:math:`2\pi` rad equals 360 degress).
 *
 *        .. math::
 *        sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L261
 * \param data The input
 * \return new symbol
 */
inline Symbol sin(Symbol data) {
  return Operator("sin")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Calculate ``log(1 + x)``
 *
 *        This function is more accurate than ``log(1 + x)``  for small ``x`` so that
 *        :math:`1+x\approx 1`
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L275
 * \param data The input
 * \return new symbol
 */
inline Symbol log1p(Symbol data) {
  return Operator("log1p")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Calculate ``exp(x) - 1``
 *
 *        This function provides greater precision than ``exp(x) - 1`` for small values
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L288
 * \param data The input
 * \return new symbol
 */
inline Symbol expm1(Symbol data) {
  return Operator("expm1")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Cosine, element-wise.
 *
 *        Then input is in radians (:math:`2\pi` rad equals 360 degress).
 *
 *        .. math::
 *        cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L304
 * \param data The input
 * \return new symbol
 */
inline Symbol cos(Symbol data) {
  return Operator("cos")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Tangent, element-wise.
 *
 *        Then input is in radians (:math:`2\pi` rad equals 360 degress).
 *
 *        .. math::
 *        tan([0, \pi/4, \pi/2]) = [0, 1, -inf]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L320
 * \param data The input
 * \return new symbol
 */
inline Symbol tan(Symbol data) {
  return Operator("tan")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Inverse sine, element-wise.
 *
 *        The input should be in range :math:`[-1, 1]`.
 *        The output is in the closed interval :math:`[-\pi/2, \pi/2]`
 *
 *        .. math::
 *        arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L337
 * \param data The input
 * \return new symbol
 */
inline Symbol arcsin(Symbol data) {
  return Operator("arcsin")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Inverse cosine, element-wise.
 *
 *        The input should be in range :math:`[-1, 1]`.
 *        The output is in the closed interval :math:`[0, \pi]`
 *
 *        .. math::
 *        arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L354
 * \param data The input
 * \return new symbol
 */
inline Symbol arccos(Symbol data) {
  return Operator("arccos")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Inverse tangent, element-wise.
 *
 *        The output is in the closed interval :math:`[-\pi/2, \pi/2]`
 *
 *        .. math::
 *        arccos([-1, 0, 1]) = [-\pi/4, 0, \pi/4]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L370
 * \param data The input
 * \return new symbol
 */
inline Symbol arctan(Symbol data) {
  return Operator("arctan")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Convert angles from radians to degrees.
 *
 *        .. math::
 *        degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L384
 * \param data The input
 * \return new symbol
 */
inline Symbol degrees(Symbol data) {
  return Operator("degrees")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Convert angles from degrees to radians.
 *
 *        .. math::
 *        radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L398
 * \param data The input
 * \return new symbol
 */
inline Symbol radians(Symbol data) {
  return Operator("radians")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Hyperbolic sine, element-wise.
 *
 *        For example::
 *        sinh(x) = 0.5\times(exp(x) - exp(-x))
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L412
 * \param data The input
 * \return new symbol
 */
inline Symbol sinh(Symbol data) {
  return Operator("sinh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Hyperbolic cosine, element-wise.
 *
 *        For example::
 *        cosh(x) = 0.5\times(exp(x) + exp(-x))
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L426
 * \param data The input
 * \return new symbol
 */
inline Symbol cosh(Symbol data) {
  return Operator("cosh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Hyperbolic tangent element-wise.
 *
 *        For example::
 *        tanh(x) = sinh(x) / cosh(x)
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L440
 * \param data The input
 * \return new symbol
 */
inline Symbol tanh(Symbol data) {
  return Operator("tanh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Inverse hyperbolic sine, element-wise.
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L450
 * \param data The input
 * \return new symbol
 */
inline Symbol arcsinh(Symbol data) {
  return Operator("arcsinh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Inverse hyperbolic cosine, element-wise.
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L460
 * \param data The input
 * \return new symbol
 */
inline Symbol arccosh(Symbol data) {
  return Operator("arccosh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Inverse hyperbolic tangent, element-wise.
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:L470
 * \param data The input
 * \return new symbol
 */
inline Symbol arctanh(Symbol data) {
  return Operator("arctanh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif The gamma function (extension of the factorial function), element-wise
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:479
 * \param data The input
 * \return new symbol
 */
inline Symbol gamma(Symbol data) {
  return Operator("gamma")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Log of the absolute value of the gamma function, element-wise
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/elemwise_unary_op.cc:488
 * \param data The input
 * \return new symbol
 */
inline Symbol gammaln(Symbol data) {
  return Operator("gammaln")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Map integer index to vector representations (embeddings). Those embeddings are
 *        learnable parameters. For a input of shape (d1, ..., dK), the output shape is
 *        (d1, ..., dK, output_dim). All the input values should be integers in the range
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/indexing_op.cc:19
 * \param data Input data to the EmbeddingOp.
 * \param weight Embedding weight matrix.
 * \param input_dim vocabulary size of the input indices.
 * \param output_dim dimension of the embedding vectors.
 * \return new symbol
 */
inline Symbol Embedding(Symbol data,
                        Symbol weight,
                        int input_dim,
                        int output_dim) {
  return Operator("Embedding")
           .SetParam("input_dim", input_dim)
           .SetParam("output_dim", output_dim)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .CreateSymbol();
}

/*!
 * \breif Take elements from an array along an axis.
 *
 *        Slice along a particular axis with the provided indices. E.g., given an input
 *        with shape ``(d0, d1, d2)`` and indices with shape ``(i0, i1)``, then the output
 *        will have shape ``(i0, i1, d1, d2)``, with::
 *
 *        output[i,j,:,:] = input[indices[i,j],:,:]
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *
 *        [[ 3.,  4.],
 *        [ 5.,  6.]]]
 *
 *        .. note::
 *        Only slicing axis 0 is supported now.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/indexing_op.cc:L79
 * \param a The source array.
 * \param indices The indices of the values to extract.
 * \param axis the axis of data tensor to be taken.
 * \param mode specify how out-of-bound indices bahave.
 * \return new symbol
 */
inline Symbol take(Symbol a,
                   Symbol indices,
                   int axis = 0,
                   TakeMode mode = TakeMode::raise) {
  static const char *TakeModeValues[] = {
    "clip",
    "raise",
    "wrap"
  };
  return Operator("take")
           .SetParam("axis", axis)
           .SetParam("mode", TakeModeValues[int(mode)])
           .SetInput("a", a)
           .SetInput("indices", indices)
           .CreateSymbol();
}

/*!
 * \breif Take elements from a data batch.
 *
 *        Given an ``(d0, d1)`` input array, and ``(d0,)`` indices, the output will be a
 *        ``(d0,)`` computed by::
 *
 *        output[i] = input[i, indices[i]]
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        batch_take(x, [0,1,0]) = [ 1.  4.  5.]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/indexing_op.cc:L131
 * \param a Input data array
 * \param indices index array
 * \return new symbol
 */
inline Symbol batch_take(Symbol a,
                         Symbol indices) {
  return Operator("batch_take")
           .SetInput("a", a)
           .SetInput("indices", indices)
           .CreateSymbol();
}

/*!
 * \breif Returns a one-hot array.
 *
 *        The locations represented by ``indices`` take value ``on_value``, while all
 *        other locations take value ``off_value``.
 *
 *        Assume ``indices`` has shape ``(i0, i1)``, then the output will have shape
 *        ``(i0, i1, depth)`` and::
 *
 *        output[i,j,:] = off_value
 *        output[i,j,indices[i,j]] = on_value
 *
 *        Examples::
 *
 *        one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]
 *        [ 1.  0.  0.]
 *        [ 0.  0.  1.]
 *        [ 1.  0.  0.]]
 *
 *        one_hot([1,0,2,0], 3, on_value=8, off_value=1,
 *        dtype='int32') = [[1 8 1]
 *        [8 1 1]
 *        [1 1 8]
 *        [8 1 1]]
 *
 *        one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]
 *        [ 1.  0.  0.]]
 *
 *        [[ 0.  1.  0.]
 *        [ 1.  0.  0.]]
 *
 *        [[ 0.  0.  1.]
 *        [ 1.  0.  0.]]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/indexing_op.cc:L177
 * \param indices array of locations where to set on_value
 * \param depth The dimension size at dim = axis.
 * \param on_value The value assigned to the locations represented by indices.
 * \param off_value The value assigned to the locations not represented by indices.
 * \param dtype DType of the output
 * \return new symbol
 */
inline Symbol one_hot(Symbol indices,
                      int depth,
                      double on_value = 1,
                      double off_value = 0,
                      One_hotDtype dtype = One_hotDtype::float32) {
  static const char *One_hotDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("one_hot")
           .SetParam("depth", depth)
           .SetParam("on_value", on_value)
           .SetParam("off_value", off_value)
           .SetParam("dtype", One_hotDtypeValues[int(dtype)])
           .SetInput("indices", indices)
           .CreateSymbol();
}

/*!
 * \breif Reshape array into a new shape.
 *
 *        The shape is a tuple of int such as (2,3,4). The new shape should not change the
 *        array size. For example::
 *
 *        reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]
 *
 *        In addition, we can use special codes, which are integers less than
 *        1, on some shape dimensions. To inference the output shape, we set it to an
 *        empty tuple at beginning. When continuously pop dimensions from the original
 *        shape starting from the beginning, and then push translated results into the
 *        shape.
 *
 *        Each special code presents a way of translation.
 *
 *        - ``0`` for copying one. Pop one input dimension and push into the output. For
 *
 *        - input=(2,3,4), shape=(4,0,2), output=(4,3,2)
 *        - input=(2,3,4), shape=(2,0,0), output=(2,3,4)
 *
 *        - ``-1`` for inference. Push a placeholder into the output whose value will be
 *
 *        - input=(2,3,4), shape=(6,1,-1), output=(6,1,4)
 *        - input=(2,3,4), shape=(3,-1,8), output=(3,1,8)
 *        - input=(2,3,4), shape=(-1,), output=(24,)
 *
 *        - ``-2`` for copying all. Pop all remaining input dimensions and push them into
 *        the output::
 *
 *        - input=(2,3,4), shape=(-2), output=(9,8,7)
 *        - input=(2,3,4), shape=(2,-2), output=(2,3,4)
 *        - input=(2,3,4), shape=(-2,1,1), output=(2,3,4,1,1)
 *
 *        - ``-3`` for merging two dimensions. Pop two input dimensions, compute the
 *        push into the output::
 *
 *        - input=(2,3,4), shape=(-3,4), output=(6,4)
 *        - input=(2,3,4), shape=(0,-3), output=(2,12)
 *        - input=(2,3,4), shape=(-3,-2), output=(6,4)
 *
 *        - ``-4`` for splitting two dimensions. Pop one input dimensions, next split it
 *        according to the next two dimensions (can contain one ``-1``) specified after
 *        this code, then push into the output::
 *
 *        - input=(2,3,4), shape=(-4,1,2,-2), output=(1,2,3,4)
 *        - input=(2,3,4), shape=(2,-4,-1,3,-2), output=(2,1,3,4)
 *
 *        If the argument ``reverse`` is set to be true, then translating the input shape
 *        from right to left. For example, with input shape (10, 5, 4) target shape (-1,
 *        0), then the output shape will be (50,4) if ``reverse=1``, otherwise it will be
 *        (40,5).
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L78
 * \param data Input data to reshape.
 * \param target_shape (Deprecated! Use ``shape`` instead.) Target new shape. One and
 * \param keep_highest (Deprecated! Use ``shape`` instead.) Whether keep the highest dim
 *        unchanged.If set to true, then the first dim in target_shape is ignored,and
 * \param shape The target shape
 * \param reverse If true then translating the input shape from right to left
 * \return new symbol
 */
inline Symbol Reshape(Symbol data,
                      Shape target_shape = Shape(0,0),
                      bool keep_highest = false,
                      Shape shape = Shape(),
                      bool reverse = false) {
  return Operator("Reshape")
           .SetParam("target_shape", target_shape)
           .SetParam("keep_highest", keep_highest)
           .SetParam("shape", shape)
           .SetParam("reverse", reverse)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Flatten input into a 2-D array by collapsing the higher dimensions.
 *
 *        Assume the input array has shape ``(d1, d2, ..., dk)``, then ``flatten``
 *        the input array into shape ``(d1, d2*...*dk)``.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L101
 * \param data Input data to reshape.
 * \return new symbol
 */
inline Symbol Flatten(Symbol data) {
  return Operator("Flatten")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Permute the dimensions of an array.
 *
 *        Examples::
 *
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *
 *        transpose(x) = [[ 1.,  3.],
 *        [ 2.,  4.]]
 *
 *        x = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *
 *        [[ 5.,  6.],
 *        [ 7.,  8.]]]
 *
 *        transpose(x) = [[[ 1.,  5.],
 *        [ 3.,  7.]],
 *
 *        [[ 2.,  6.],
 *        [ 4.,  8.]]]
 *
 *        transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
 *        [ 5.,  6.]],
 *
 *        [[ 3.,  4.],
 *        [ 7.,  8.]]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L142
 * \param data Source input
 * \param axes Target axis order. By default the axes will be inverted.
 * \return new symbol
 */
inline Symbol transpose(Symbol data,
                        Shape axes = Shape()) {
  return Operator("transpose")
           .SetParam("axes", axes)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Insert a new axis with size 1 into the array shape
 *
 *        For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``
 *        will return a new array with shape ``(2,1,3,4)``.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L175
 * \param data Source input
 * \param axis Position (amongst axes) where new axis is to be inserted.
 * \return new symbol
 */
inline Symbol expand_dims(Symbol data,
                          uint32_t axis) {
  return Operator("expand_dims")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Crop a continuous region from the array.
 *
 *        Assume the input array has *n* dimensions, given ``begin=(b_1, ..., b_n)`` and
 *        ``end=(e_1, ..., e_n)``, then ``crop`` will return a region with shape
 *        ``(e_1-b_1, ..., e_n-b_n)``. The result's *k*-th dimension contains elements
 *        from the *k*-th dimension of the input array with the open range ``[b_k, e_k)``.
 *
 *        For example::
 *
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        crop(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
 *        [ 6.,  7.,  8.]]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L207
 * \param data Source input
 * \param begin starting coordinates
 * \param end ending coordinates
 * \return new symbol
 */
inline Symbol slice(Symbol data,
                    Shape begin,
                    Shape end) {
  return Operator("slice")
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Slice along a given axis.
 *
 *        Examples:
 *
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],
 *        [  5.,   6.],
 *        [  9.,  10.]]
 *
 *        slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],
 *        [  6.,   7.],
 *        [ 10.,  11.]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L285
 * \param data Source input
 * \param axis The axis to be sliced. Negative axis means to count from the last to the
 * \param begin The beginning index to be sliced. Negative values are interpreted as
 * \param end The end index to be sliced. The end can be None, in which case all the rest
 *        elements are used. Also, negative values are interpreted as counting from the
 * \return new symbol
 */
inline Symbol slice_axis(Symbol data,
                         int axis,
                         int begin,
                         dmlc::optional<int> end) {
  return Operator("slice_axis")
           .SetParam("axis", axis)
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Dot product of two arrays.
 *
 *        ``dot``'s behavior depends on the input array dimensions:
 *
 *        - 1-D arrays: inner product of vectors
 *        - 2-D arrays: matrix multiplication
 *        - N-D arrays: a sum product over the last axis of the first input and the first
 *        axis of the second input
 *
 *        For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape
 *        result array will have shape `(n,m,r,s)`. It is computed by::
 *
 *        dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L318
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \return new symbol
 */
inline Symbol dot(Symbol lhs,
                  Symbol rhs,
                  bool transpose_a = false,
                  bool transpose_b = false) {
  return Operator("dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Batchwise dot product.
 *
 *        ``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
 *        ``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.
 *
 *        For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape
 *        `(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,
 *        which is computed by::
 *
 *        batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L354
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \return new symbol
 */
inline Symbol batch_dot(Symbol lhs,
                        Symbol rhs,
                        bool transpose_a = false,
                        bool transpose_b = false) {
  return Operator("batch_dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Clip (limit) the values in an array, elementwise
 *
 *        Given an interval, values outside the interval are clipped to the interval
 *        edges. That is::
 *
 *        clip(x) = max(min(x, a_max)), a_min)
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L393
 * \param data Source input
 * \param a_min Minimum value
 * \param a_max Maximum value
 * \return new symbol
 */
inline Symbol clip(Symbol data,
                   mx_float a_min,
                   mx_float a_max) {
  return Operator("clip")
           .SetParam("a_min", a_min)
           .SetParam("a_max", a_max)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Repeat elements of an array.
 *
 *        In default, ``repeat`` flatten the input array into 1-D and then repeat the
 *        elements::
 *
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *
 *        repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]
 *
 *        We can also choose a particular axis to repeat, in which a negative axis is
 *        interpreted counting from the backward::
 *
 *        repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *
 *        repeat(x, repeats=2, axis=-1) = [[ 1.,  2.],
 *        [ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 3.,  4.]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L432
 * \param data Input data array
 * \param repeats The number of repetitions for each element.
 * \param axis The axis along which to repeat values. The negative numbers are
 *        interpreted counting from the backward. By default, use the flattened input
 * \return new symbol
 */
inline Symbol repeat(Symbol data,
                     int repeats,
                     dmlc::optional<int> axis = dmlc::optional<int>()) {
  return Operator("repeat")
           .SetParam("repeats", repeats)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Repeat the whole array by multiple times.
 *
 *        If ``reps`` has length *d*, and input array has dimension of *n*. There are
 *        there cases:
 *
 *        - **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::
 *
 *        x = [[1, 2],
 *        [3, 4]]
 *
 *        tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]
 *
 *        - **n>d**. ``reps`` is promoted to length *n* by pre-pending 1’s to it. Thus
 *        an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::
 *
 *
 *        tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.]]
 *
 *        - **n<d**. The input is promoted to be d-dimensional by prepending new axes. So
 *        shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::
 *
 *        tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]],
 *
 *        [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:L489
 * \param data Input data array
 * \param reps The number of times for repeating the tensor a. If reps has length d, the
 *        result will have dimension of max(d, a.ndim); If a.ndim < d, a is promoted to
 *        be d-dimensional by prepending new axes. If a.ndim > d, reps is promoted to
 * \return new symbol
 */
inline Symbol tile(Symbol data,
                   Shape reps) {
  return Operator("tile")
           .SetParam("reps", reps)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Reverse elements of an array with axis
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/matrix_op.cc:512
 * \param data Input data array
 * \param axis The axis which to reverse elements.
 * \return new symbol
 */
inline Symbol reverse(Symbol data,
                      Shape axis) {
  return Operator("reverse")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Draw samples from a uniform distribution.
 *
 *        Samples are uniformly distributed over the half-open interval [low, high)
 *        (includes low, but excludes high)::
 *
 *        nd.uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
 *        [ 0.54488319,  0.84725171]]
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/sample_op.cc:L24
 * \param low The lower bound of distribution
 * \param high The upper bound of distribution
 * \param shape The shape of the output
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for
 * \param dtype DType of the output. If output given, set to type of output.If output not
 * \return new symbol
 */
inline Symbol uniform(mx_float low = 0,
                      mx_float high = 1,
                      Shape shape = Shape(),
                      const std::string& ctx = "",
                      UniformDtype dtype = UniformDtype::None) {
  static const char *UniformDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("uniform")
           .SetParam("low", low)
           .SetParam("high", high)
           .SetParam("shape", shape)
           .SetParam("dtype", UniformDtypeValues[int(dtype)])
           .CreateSymbol();
}

/*!
 * \breif Draw random samples from a normal (Gaussian) distribution.
 *
 *        Examples::
 *
 *        normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
 *        [-1.23474145,  1.55807114]]
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/sample_op.cc:L35
 * \param loc Mean of the distribution.
 * \param scale Standard deviation of the distribution.
 * \param shape The shape of the output
 * \param ctx Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for
 * \param dtype DType of the output. If output given, set to type of output.If output not
 * \return new symbol
 */
inline Symbol normal(mx_float loc = 0,
                     mx_float scale = 1,
                     Shape shape = Shape(),
                     const std::string& ctx = "",
                     NormalDtype dtype = NormalDtype::None) {
  static const char *NormalDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("normal")
           .SetParam("loc", loc)
           .SetParam("scale", scale)
           .SetParam("shape", shape)
           .SetParam("dtype", NormalDtypeValues[int(dtype)])
           .CreateSymbol();
}

/*!
 * \breif Returns the indices of the maximum values along an axis.
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/broadcast_reduce_op_index.cc:11
 * \param data The input
 * \param axis Empty or unsigned. The axis to perform the reduction.If left empty, a
 * \param keepdims If true, the axis which is reduced is left in the result as dimension
 * \return new symbol
 */
inline Symbol argmax(Symbol data,
                     int axis = -1,
                     bool keepdims = false) {
  return Operator("argmax")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the indices of the minimum values along an axis.
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/broadcast_reduce_op_index.cc:16
 * \param data The input
 * \param axis Empty or unsigned. The axis to perform the reduction.If left empty, a
 * \param keepdims If true, the axis which is reduced is left in the result as dimension
 * \return new symbol
 */
inline Symbol argmin(Symbol data,
                     int axis = -1,
                     bool keepdims = false) {
  return Operator("argmin")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif
 * \param src Source input
 * \return new symbol
 */
inline Symbol argmax_channel(Symbol src) {
  return Operator("argmax_channel")
           .SetInput("src", src)
           .CreateSymbol();
}

/*!
 * \breif Compute the sum of array elements over given axes.
 *
 *        The argument ``axis`` specifies the axes to compute over:
 *
 *        - **()**: compute over all elements into a scalar array with shape ``(1,)``.
 *        the default option.
 *        - **int**: compute over along a particular axis. If input has shape ``(n, m,
 *        use ``axis=0`` will result in an array with shape ``(m, k)``.
 *        - **tuple of int**: compute over multiple axes. Again assume input shape ``(n,
 *        k)``, with ``axis=(0,2)`` we obtain a ``(m,)`` shape array.
 *
 *        If ``keepdims = 1``, then the result array will has the same number of
 *        as the input, while the reduced axes will have size 1.
 *
 *
 *        Defined in
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol sum(Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false) {
  return Operator("sum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Compute the mean of array elements over given axes.
 *
 *        The argument ``axis`` specifies the axes to compute over:
 *
 *        - **()**: compute over all elements into a scalar array with shape ``(1,)``.
 *        the default option.
 *        - **int**: compute over along a particular axis. If input has shape ``(n, m,
 *        use ``axis=0`` will result in an array with shape ``(m, k)``.
 *        - **tuple of int**: compute over multiple axes. Again assume input shape ``(n,
 *        k)``, with ``axis=(0,2)`` we obtain a ``(m,)`` shape array.
 *
 *        If ``keepdims = 1``, then the result array will has the same number of
 *        as the input, while the reduced axes will have size 1.
 *
 *
 *        Defined in
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol mean(Symbol data,
                   Shape axis = Shape(),
                   bool keepdims = false) {
  return Operator("mean")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Compute the product of array elements over given axes.
 *
 *        The argument ``axis`` specifies the axes to compute over:
 *
 *        - **()**: compute over all elements into a scalar array with shape ``(1,)``.
 *        the default option.
 *        - **int**: compute over along a particular axis. If input has shape ``(n, m,
 *        use ``axis=0`` will result in an array with shape ``(m, k)``.
 *        - **tuple of int**: compute over multiple axes. Again assume input shape ``(n,
 *        k)``, with ``axis=(0,2)`` we obtain a ``(m,)`` shape array.
 *
 *        If ``keepdims = 1``, then the result array will has the same number of
 *        as the input, while the reduced axes will have size 1.
 *
 *
 *        Defined in
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol prod(Symbol data,
                   Shape axis = Shape(),
                   bool keepdims = false) {
  return Operator("prod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Compute the sum of array elements over given axes with ``NaN`` ignored
 *
 *        Refer to ``sum`` for more details.
 *
 *
 *
 *        Defined in
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol nansum(Symbol data,
                     Shape axis = Shape(),
                     bool keepdims = false) {
  return Operator("nansum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Compute the product of array elements over given axes with ``NaN`` ignored
 *
 *        Refer to ``prod`` for more details.
 *
 *
 *
 *        Defined in
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol nanprod(Symbol data,
                      Shape axis = Shape(),
                      bool keepdims = false) {
  return Operator("nanprod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Compute the max of array elements over given axes.
 *
 *        The argument ``axis`` specifies the axes to compute over:
 *
 *        - **()**: compute over all elements into a scalar array with shape ``(1,)``.
 *        the default option.
 *        - **int**: compute over along a particular axis. If input has shape ``(n, m,
 *        use ``axis=0`` will result in an array with shape ``(m, k)``.
 *        - **tuple of int**: compute over multiple axes. Again assume input shape ``(n,
 *        k)``, with ``axis=(0,2)`` we obtain a ``(m,)`` shape array.
 *
 *        If ``keepdims = 1``, then the result array will has the same number of
 *        as the input, while the reduced axes will have size 1.
 *
 *
 *        Defined in
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol max(Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false) {
  return Operator("max")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Compute the min of array elements over given axes.
 *
 *        The argument ``axis`` specifies the axes to compute over:
 *
 *        - **()**: compute over all elements into a scalar array with shape ``(1,)``.
 *        the default option.
 *        - **int**: compute over along a particular axis. If input has shape ``(n, m,
 *        use ``axis=0`` will result in an array with shape ``(m, k)``.
 *        - **tuple of int**: compute over multiple axes. Again assume input shape ``(n,
 *        k)``, with ``axis=(0,2)`` we obtain a ``(m,)`` shape array.
 *
 *        If ``keepdims = 1``, then the result array will has the same number of
 *        as the input, while the reduced axes will have size 1.
 *
 *
 *        Defined in
 * \param data The input
 * \param axis The axes to perform the reduction.
 * \param keepdims If true, the axes which are reduced are left in the result as
 * \return new symbol
 */
inline Symbol min(Symbol data,
                  Shape axis = Shape(),
                  bool keepdims = false) {
  return Operator("min")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Broadcast an array over particular axes.
 *
 *        Broadcasting is allowed on axes which size 1, such as from ``(2,1,3,1)`` to
 *        ``(2,8,3,9)``. Elemenets will be duplicated on the broadcasted axes.
 *
 *        For example::
 *
 *        // given (1,2,1) shape x
 *        x = [[[ 1.],
 *        [ 2.]]]
 *
 *        // broadcast on axis 2
 *        broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]]
 *        // broadcast on axes 0 and 2
 *        broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]],
 *        [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]]
 *
 *
 *        Defined in
 * \param data The input
 * \param axis The axes to perform the broadcasting.
 * \param size Target sizes of the broadcasting axes.
 * \return new symbol
 */
inline Symbol broadcast_axis(Symbol data,
                             Shape axis = Shape(),
                             Shape size = Shape()) {
  return Operator("broadcast_axis")
           .SetParam("axis", axis)
           .SetParam("size", size)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Broadcast an array to a new shape.
 *
 *        Broadcasting is allowed on axes which size 1, such as from ``(2,1,3,1)`` to
 *        ``(2,8,3,9)``. Elemenets will be duplicated on the broadcasted axes.
 *
 *        For example::
 *
 *        broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
 *        [ 1.,  2.,  3.]])
 *
 *        The dimensions that will not be changed can also use the special code ``0`` that
 *        means copy the original value. So with ``shape=(2,0)`` we will obtain the same
 *        results in the above example.
 *
 *
 *
 *        Defined in
 * \param data The input
 * \param shape The shape of the desired array. We can set the dim to zero if it's same
 *        as the original. E.g `A = broadcast_to(B, shape=(10, 0, 0))` has the same
 * \return new symbol
 */
inline Symbol broadcast_to(Symbol data,
                           Shape shape = Shape()) {
  return Operator("broadcast_to")
           .SetParam("shape", shape)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Compute the L2 norm.
 *
 *        Flatten then input array and then compute the l2 norm.
 *
 *        Examples::
 *
 *        x = [[1, 2],
 *        [3, 4]]
 *
 *        norm(x) = [5.47722578]
 *
 *
 *
 *        Defined in
 * \param src Source input
 * \return new symbol
 */
inline Symbol norm(Symbol src) {
  return Operator("norm")
           .SetInput("src", src)
           .CreateSymbol();
}

/*!
 * \breif Given three ndarrays, condition, x, and y, return an ndarray with the elements
 *        from x or y, depending on the elements from condition are true or false. x and
 *        y must have the same shape. If condition has the same shape as x, each element
 *        in the output array is from x if the corresponding element in the condition is
 *        true, and from y if false. If condtion does not have the same shape as x, it
 *        must be a 1D array whose size is the same as x's first dimension size. Each row
 *        of the output array is from x's row if the corresponding element from condition
 *
 *        From:/home/xlidc/mxnet/src/operator/tensor/control_flow_op.cc:21
 * \param condition condition array
 * \param x
 * \param y
 * \return new symbol
 */
inline Symbol where(Symbol condition,
                    Symbol x,
                    Symbol y) {
  return Operator("where")
           .SetInput("condition", condition)
           .SetInput("x", x)
           .SetInput("y", y)
           .CreateSymbol();
}

/*!
 * \breif Add arguments, element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_add(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Substract arguments, element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_sub(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_sub")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Multiply arguments, element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_mul(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Divide arguments, element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_div(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif First array elements raised to powers from second array,
 *        element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 *        /home/xlidc/mxnet/src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L16
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_power(Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_power")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Element-wise maximum of array elements with broadcasting.
 *
 *
 *
 *        Defined in
 *        /home/xlidc/mxnet/src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L34
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_maximum(Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_maximum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Element-wise minimum of array elements with broadcasting.
 *
 *
 *
 *        Defined in
 *        /home/xlidc/mxnet/src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L52
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_minimum(Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_minimum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Given the "legs" of a right triangle, return its hypotenuse
 *        with broadcasting.
 *
 *
 *
 *        Defined in
 *        /home/xlidc/mxnet/src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L71
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_hypot(Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_hypot")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Return (lhs == rhs), element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_equal(Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Return (lhs != rhs), element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_not_equal(Symbol lhs,
                                  Symbol rhs) {
  return Operator("broadcast_not_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Return (lhs > rhs), element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_greater(Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_greater")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Return (lhs >= rhs), element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_greater_equal(Symbol lhs,
                                      Symbol rhs) {
  return Operator("broadcast_greater_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Return (lhs < rhs), element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_lesser(Symbol lhs,
                               Symbol rhs) {
  return Operator("broadcast_lesser")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Return (lhs <= rhs), element-wise with broadcasting.
 *
 *
 *
 *        Defined in
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol broadcast_lesser_equal(Symbol lhs,
                                     Symbol rhs) {
  return Operator("broadcast_lesser_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Add all input arguments element-wise.
 *
 *        .. math::
 *        add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n
 *
 *        ``add_n`` is potentially more efficient than calling ``add`` by `n` times.
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/tensor/elemwise_sum.cc:L63
 * \param args Positional input arguments
 * \return new symbol
 */
inline Symbol add_n(const std::vector<Symbol>& args) {
  return Operator("add_n")
(args)
           .CreateSymbol();
}

/*!
 * \breif Custom operator implemented in frontend.
 * \param op_type Type of custom operator. Must be registered first.
 * \return new symbol
 */
inline Symbol Custom(const std::string& op_type) {
  return Operator("Custom")
           .CreateSymbol();
}

/*!
 * \breif Elementwise activation function.
 *        The activation operations are applied elementwisely to each array elements. The
 *        following types are supported:
 *
 *        - `relu`: Rectified Linear Unit, `y = max(x, 0)`
 *        - `sigmoid`: `y = 1 / (1 + exp(-x))`
 *        - `tanh`: Hyperbolic tangent, `y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
 *        - `softrelu`: Soft ReLU, or SoftPlus, `y = log(1 + exp(x))`
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/activation.cc:L76
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \return new symbol
 */
inline Symbol Activation(Symbol data,
                         ActivationActType act_type) {
  static const char *ActivationActTypeValues[] = {
    "relu",
    "sigmoid",
    "softrelu",
    "tanh"
  };
  return Operator("Activation")
           .SetParam("act_type", ActivationActTypeValues[int(act_type)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply bilinear sampling to input feature map, which is the key of "[NIPS2015]
 *        output[batch, channel, y_dst, x_dst] = G(data[batch, channel, y_src, x_src)
 *        x_dst, y_dst enumerate all spatial locations in output
 *        x_src = grid[batch, 0, y_dst, x_dst]
 *        y_src = grid[batch, 1, y_dst, x_dst]
 *        G() denotes the bilinear interpolation kernel
 *        The out-boundary points will be padded as zeros. (The boundary is defined to be
 *        The shape of output will be (data.shape[0], data.shape[1], grid.shape[2],
 *        The operator assumes that grid has been nomalized. If you want to design a
 * \param data Input data to the BilinearsamplerOp.
 * \param grid Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src
 * \return new symbol
 */
inline Symbol BilinearSampler(Symbol data,
                              Symbol grid) {
  return Operator("BilinearSampler")
           .SetInput("data", data)
           .SetInput("grid", grid)
           .CreateSymbol();
}

/*!
 * \breif Compute *N*-D convolution on *(N+2)*-D input.
 *
 *        In the simplest 2-D convolution, given input data with shape *(batch_size,
 *        channel, height, weight)*, the output is computed by
 *
 *        .. math::
 *
 *        out[n,i,:,:] = bias[i] + \sum_{j=0}^{num\_filter} data[n,j,:,:] \star
 *        weight[i,j,:,:]
 *
 *        where :math:`\star` is the 2-D cross-correlation operator.
 *
 *        For general 2-D convolution, the shapes are
 *
 *        - **data**: *(batch_size, channel, height, weight)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_height, out_weight)*.
 *
 *        Define::
 *
 *        f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1
 *
 *        then we have::
 *
 *        out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
 *        out_weight=f(weight, kernel[1], pad[1], stride[1], dilate[1])
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *        The default data ``layout`` is *NCHW*, namely *(batch_size, channle, height,
 *        weight)*. We can choose other layouts such as *NHWC*.
 *
 *        If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
 *        evenly into *g* parts along the channel axis, and also evenly split ``weight``
 *        along the first dimension. Next compute the convolution on the *i*-th part of
 *        the data with the *i*-th weight part. The output is obtained by concating all
 *        the *g* results.
 *
 *        To perform 1-D convolution, simply use 2-D convolution but set the last axis
 *        size to be 1 for both data and weight.
 *
 *        3-D convolution adds an additional depth dimension besides height and
 *        weight. The shapes are
 *
 *        - **data**: *(batch_size, channel, depth, height, weight)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_depth, out_height, out_weight)*.
 *
 *        Both ``weight`` and ``bias`` are learnable parameters.
 *
 *        There are other options to tune the performance.
 *
 *        - **cudnn_tune**: enable this option leads to higher startup time but may give
 *        faster speed. Options are
 *
 *        - **off**: no tuning
 *        - **limited_workspace**:run test and pick the fastest algorithm that doesn't
 *        exceed workspace limit.
 *        - **fastest**: pick the fastest algorithm and ignore workspace limit.
 *        - **None** (default): the behavior is determined by environment variable
 *        ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
 *        (default), 2 for fastest.
 *
 *        - **workspace**: A large number leads to more (GPU) memory usage but may improve
 *        the performance.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/convolution.cc:L143
 * \param data Input data to the ConvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions.
 * \param workspace Maximum temperal workspace allowed for convolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution(Symbol data,
                          Symbol weight,
                          Symbol bias,
                          Shape kernel,
                          uint32_t num_filter,
                          Shape stride = Shape(),
                          Shape dilate = Shape(),
                          Shape pad = Shape(),
                          uint32_t num_group = 1,
                          uint64_t workspace = 1024,
                          bool no_bias = false,
                          ConvolutionCudnnTune cudnn_tune = ConvolutionCudnnTune::None,
                          bool cudnn_off = false,
                          ConvolutionLayout layout = ConvolutionLayout::None) {
  static const char *ConvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *ConvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", ConvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", ConvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif Apply correlation to inputs
 * \param data1 Input data1 to the correlation.
 * \param data2 Input data2 to the correlation.
 * \param kernel_size kernel size for Correlation must be an odd number
 * \param max_displacement Max displacement of Correlation
 * \param stride1 stride1 quantize data1 globally
 * \param stride2 stride2 quantize data2 within the neighborhood centered around data1
 * \param pad_size pad for Correlation
 * \param is_multiply operation type is either multiplication or subduction
 * \return new symbol
 */
inline Symbol Correlation(Symbol data1,
                          Symbol data2,
                          uint32_t kernel_size = 1,
                          uint32_t max_displacement = 1,
                          uint32_t stride1 = 1,
                          uint32_t stride2 = 1,
                          uint32_t pad_size = 0,
                          bool is_multiply = true) {
  return Operator("Correlation")
           .SetParam("kernel_size", kernel_size)
           .SetParam("max_displacement", max_displacement)
           .SetParam("stride1", stride1)
           .SetParam("stride2", stride2)
           .SetParam("pad_size", pad_size)
           .SetParam("is_multiply", is_multiply)
           .SetInput("data1", data1)
           .SetInput("data2", data2)
           .CreateSymbol();
}

/*!
 * \breif Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or
 *        with width and height of the second input symbol, i.e., with one input, we need
 *        h_w to specify the crop height and width, otherwise the second input symbol's
 * \param data Tensor or List of Tensors, the second input will be used as crop_like
 * \param num_args Number of inputs for crop, if equals one, then we will use the h_wfor
 *        crop height and width, else if equals two, then we will use the heightand width
 * \param offset crop offset coordinate: (y, x)
 * \param h_w crop height and weight: (h, w)
 * \param center_crop If set to true, then it will use be the center_crop,or it will crop
 * \return new symbol
 */
inline Symbol Crop(const std::vector<Symbol>& data,
                   int num_args,
                   Shape offset = Shape(0,0),
                   Shape h_w = Shape(0,0),
                   bool center_crop = false) {
  return Operator("Crop")
           .SetParam("num_args", num_args)
           .SetParam("offset", offset)
           .SetParam("h_w", h_w)
           .SetParam("center_crop", center_crop)
(data)
           .CreateSymbol();
}

/*!
 * \breif Apply deconvolution to input then add a bias.
 * \param data Input data to the DeconvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel deconvolution kernel size: (y, x)
 * \param num_filter deconvolution filter(channel) number
 * \param stride deconvolution stride: (y, x)
 * \param pad pad for deconvolution: (y, x), a good number is : (kernel-1)/2, if
 * \param adj adjustment for output shape: (y, x), if target_shape set, adj will be
 * \param target_shape output shape with targe shape : (y, x)
 * \param num_group number of groups partition
 * \param workspace Tmp workspace for deconvolution (MB)
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
inline Symbol Deconvolution(Symbol data,
                            Symbol weight,
                            Symbol bias,
                            Shape kernel,
                            uint32_t num_filter,
                            Shape stride = Shape(1,1),
                            Shape pad = Shape(0,0),
                            Shape adj = Shape(0,0),
                            Shape target_shape = Shape(0,0),
                            uint32_t num_group = 1,
                            uint64_t workspace = 512,
                            bool no_bias = true) {
  return Operator("Deconvolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetParam("adj", adj)
           .SetParam("target_shape", target_shape)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif Apply dropout to input.
 *        During training, each element of the input is randomly set to zero with
 *        And then the whole tensor is rescaled by 1/(1-p) to keep the expectation the
 *        before applying dropout. During the test time, this behaves as an identity map.
 *
 * \param data Input data to dropout.
 * \param p Fraction of the input that gets dropped out at training time
 * \return new symbol
 */
inline Symbol Dropout(Symbol data,
                      mx_float p = 0.5) {
  return Operator("Dropout")
           .SetParam("p", p)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply a linear transformation: :math:`Y = XW^T + b`.
 *
 *        Shapes:
 *
 *        - **data**: `(batch_size, input_dim)`
 *        - **weight**: `(num_hidden, input_dim)`
 *        - **bias**: `(num_hidden,)`
 *        - **out**: `(batch_size, num_hidden)`
 *
 *        The learnable parameters include both ``weight`` and ``bias``.
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/fully_connected.cc:L94
 * \param data Input data.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param num_hidden Number of hidden nodes of the output.
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
inline Symbol FullyConnected(Symbol data,
                             Symbol weight,
                             Symbol bias,
                             int num_hidden,
                             bool no_bias = false) {
  return Operator("FullyConnected")
           .SetParam("num_hidden", num_hidden)
           .SetParam("no_bias", no_bias)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif An operator taking in a n-dimensional input tensor (n > 2), and normalizing the
 *        input by subtracting the mean and variance calculated over the spatial
 *        dimensions. This is an implemention of the operator described in "Instance
 *        Normalization: The Missing Ingredient for Fast Stylization", D. Ulyanov, A.
 *        Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2). This layer is similar to
 *        batch normalization, with two differences: first, the normalization is carried
 *        out per example ('instance'), not over a batch. Second, the same normalization
 *        is applied both at test and train time. This operation is also known as
 * \param data A n-dimensional tensor (n > 2) of the form [batch, channel, spatial_dim1,
 * \param gamma A vector of length 'channel', which multiplies the normalized input.
 * \param beta A vector of length 'channel', which is added to the product of the
 * \param eps Epsilon to prevent division by 0.
 * \return new symbol
 */
inline Symbol InstanceNorm(Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.001) {
  return Operator("InstanceNorm")
           .SetParam("eps", eps)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol();
}

/*!
 * \breif Set the l2 norm of each instance to a constant.
 * \param data Input data to the L2NormalizationOp.
 * \param eps Epsilon to prevent div 0
 * \param mode Normalization Mode. If set to instance, this operator will compute a norm
 *        for each instance in the batch; this is the default mode. If set to channel,
 *        this operator will compute a cross channel norm at each position of each
 * \return new symbol
 */
inline Symbol L2Normalization(Symbol data,
                              mx_float eps = 1e-10,
                              L2NormalizationMode mode = L2NormalizationMode::instance) {
  static const char *L2NormalizationModeValues[] = {
    "channel",
    "instance",
    "spatial"
  };
  return Operator("L2Normalization")
           .SetParam("eps", eps)
           .SetParam("mode", L2NormalizationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply convolution to input then add a bias.
 * \param data Input data to the ConvolutionOp.
 * \param nsize normalization window width in elements.
 * \param alpha value of the alpha variance scaling parameter in the normalization formula
 * \param beta value of the beta power parameter in the normalization formula
 * \param knorm value of the k parameter in normalization formula
 * \return new symbol
 */
inline Symbol LRN(Symbol data,
                  uint32_t nsize,
                  mx_float alpha = 0.0001,
                  mx_float beta = 0.75,
                  mx_float knorm = 2) {
  return Operator("LRN")
           .SetParam("nsize", nsize)
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetParam("knorm", knorm)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Get output from a symbol and pass 1 gradient back. This is used as a terminal
 *        loss if unary and binary operator are used to composite a loss with no
 * \param data Input data.
 * \param grad_scale gradient scale as a supplement to unary and binary operators
 * \param valid_thresh regard element valid when x > valid_thresh, this is used only in
 * \param normalization If set to null, op will not normalize on output gradient.If set
 *        to batch, op will normalize gradient by divide batch size.If set to valid, op
 * \return new symbol
 */
inline Symbol MakeLoss(Symbol data,
                       mx_float grad_scale = 1,
                       mx_float valid_thresh = 0,
                       MakeLossNormalization normalization = MakeLossNormalization::null) {
  static const char *MakeLossNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("MakeLoss")
           .SetParam("grad_scale", grad_scale)
           .SetParam("valid_thresh", valid_thresh)
           .SetParam("normalization", MakeLossNormalizationValues[int(normalization)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Perform pooling on the input.
 *
 *        The shapes for 2-D pooling is
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The defintion of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor(x+2*p-k)/s+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil(x+2*p-k)/s+1
 *
 *        But ``global_pool`` is set to be true, then do a global pooling, namely reset
 *        ``kernel=(height, width)``.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *
 *        1-D pooling is special case of 2-D pooling with *weight=1* and
 *        *kernel[1]=1*.
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data will have shape *(batch_size, channel, depth,
 *        height, width)*.
 *
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/pooling.cc:L122
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param pooling_convention Pooling convention to be applied.
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling(Symbol data,
                      Shape kernel,
                      PoolingPoolType pool_type,
                      bool global_pool = false,
                      PoolingPoolingConvention pooling_convention = PoolingPoolingConvention::valid,
                      Shape stride = Shape(),
                      Shape pad = Shape()) {
  static const char *PoolingPoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *PoolingPoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", PoolingPoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("pooling_convention", PoolingPoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Use linear regression for final output, this is used on final output of a net.
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LinearRegressionOutput(Symbol data,
                                     Symbol label,
                                     mx_float grad_scale = 1) {
  return Operator("LinearRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Use mean absolute error regression for final output, this is used on final
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol MAERegressionOutput(Symbol data,
                                  Symbol label,
                                  mx_float grad_scale = 1) {
  return Operator("MAERegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Use Logistic regression for final output, this is used on final output of a net.
 *        Logistic regression is suitable for binary classification or probability
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LogisticRegressionOutput(Symbol data,
                                       Symbol label,
                                       mx_float grad_scale = 1) {
  return Operator("LogisticRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Apply a recurrent layer to input.
 * \param data Input data to RNN
 * \param parameters Vector of all RNN trainable parameters concatenated
 * \param state initial hidden state of the RNN
 * \param state_cell initial cell state for LSTM networks (only for LSTM)
 * \param state_size size of the state for each layer
 * \param num_layers number of stacked layers
 * \param mode the type of RNN to compute
 * \param bidirectional whether to use bidirectional recurrent layers
 * \param p Dropout probability, fraction of the input that gets dropped out at training
 * \param state_outputs Whether to have the states as symbol outputs.
 * \return new symbol
 */
inline Symbol RNN(Symbol data,
                  Symbol parameters,
                  Symbol state,
                  Symbol state_cell,
                  uint32_t state_size,
                  uint32_t num_layers,
                  RNNMode mode,
                  bool bidirectional = false,
                  mx_float p = 0,
                  bool state_outputs = false) {
  static const char *RNNModeValues[] = {
    "gru",
    "lstm",
    "rnn_relu",
    "rnn_tanh"
  };
  return Operator("RNN")
           .SetParam("state_size", state_size)
           .SetParam("num_layers", num_layers)
           .SetParam("mode", RNNModeValues[int(mode)])
           .SetParam("bidirectional", bidirectional)
           .SetParam("p", p)
           .SetParam("state_outputs", state_outputs)
           .SetInput("data", data)
           .SetInput("parameters", parameters)
           .SetInput("state", state)
           .SetInput("state_cell", state_cell)
           .CreateSymbol();
}

/*!
 * \breif Performs region-of-interest pooling on inputs. Resize bounding box coordinates
 *        by spatial_scale and crop input feature maps accordingly. The cropped feature
 *        maps are pooled by max pooling to a fixed size output indicated by pooled_size.
 * \param data Input data to the pooling operator, a 4D Feature maps
 * \param rois Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]].
 *        (x1, y1) and (x2, y2) are top left and down right corners of designated region
 *        of interest. batch_index indicates the index of corresponding image in the
 * \param pooled_size fix pooled size: (h, w)
 * \param spatial_scale Ratio of input feature map height (or w) to raw image height (or
 * \return new symbol
 */
inline Symbol ROIPooling(Symbol data,
                         Symbol rois,
                         Shape pooled_size,
                         mx_float spatial_scale) {
  return Operator("ROIPooling")
           .SetParam("pooled_size", pooled_size)
           .SetParam("spatial_scale", spatial_scale)
           .SetInput("data", data)
           .SetInput("rois", rois)
           .CreateSymbol();
}

/*!
 * \breif Takes the last element of a sequence. Takes an n-dimensional tensor of the form
 *        [max sequence length, batchsize, other dims] and returns a (n-1)-dimensional
 *        tensor of the form [batchsize, other dims]. This operator takes an optional
 *        input tensor sequence_length of positive ints of dimension [batchsize] when the
 *        sequence_length option is set to true. This allows the operator to handle
 *        variable-length sequences. If sequence_length is false, then each example in
 * \param data n-dimensional input tensor of the form [max sequence length, batchsize,
 * \param sequence_length vector of sequence lengths of size batchsize
 * \param use_sequence_length If set to true, this layer takes in extra input
 * \return new symbol
 */
inline Symbol SequenceLast(Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false) {
  return Operator("SequenceLast")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \breif Sets all elements outside the sequence to a constant value. Takes an
 *        n-dimensional tensor of the form [max sequence length, batchsize, other dims]
 *        and returns a tensor of the same shape. This operator takes an optional input
 *        tensor sequence_length of positive ints of dimension [batchsize] when the
 *        sequence_length option is set to true. This allows the operator to handle
 *        variable-length sequences. If sequence_length is false, then each example in
 *        the batch is assumed to have the max sequence length, and this operator becomes
 * \param data n-dimensional input tensor of the form [max sequence length, batchsize,
 * \param sequence_length vector of sequence lengths of size batchsize
 * \param use_sequence_length If set to true, this layer takes in extra input
 * \param value The value to be used as a mask.
 * \return new symbol
 */
inline Symbol SequenceMask(Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false,
                           mx_float value = 0) {
  return Operator("SequenceMask")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("value", value)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \breif Reverses the elements of each sequence. Takes an n-dimensional tensor of the
 *        form [max sequence length, batchsize, other dims] and returns a tensor of the
 *        same shape. This operator takes an optional input tensor sequence_length of
 *        positive ints of dimension [batchsize] when the sequence_length option is set
 *        to true. This allows the operator to handle variable-length sequences. If
 *        sequence_length is false, then each example in the batch is assumed to have the
 * \param data n-dimensional input tensor of the form [max sequence length, batchsize,
 * \param sequence_length vector of sequence lengths of size batchsize
 * \param use_sequence_length If set to true, this layer takes in extra input
 * \return new symbol
 */
inline Symbol SequenceReverse(Symbol data,
                              Symbol sequence_length,
                              bool use_sequence_length = false) {
  return Operator("SequenceReverse")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \breif Apply softmax activation to input. This is intended for internal layers. For
 *        output (loss layer) please use SoftmaxOutput. If mode=instance, this operator
 *        will compute a softmax for each instance in the batch; this is the default
 *        mode. If mode=channel, this operator will compute a num_channel-class softmax
 *        at each position of each instance; this can be used for fully convolutional
 * \param data Input data to activation function.
 * \param mode Softmax Mode. If set to instance, this operator will compute a softmax for
 *        each instance in the batch; this is the default mode. If set to channel, this
 *        operator will compute a num_channel-class softmax at each position of each
 *        instance; this can be used for fully convolutional network, image segmentation,
 * \return new symbol
 */
inline Symbol SoftmaxActivation(Symbol data,
                                SoftmaxActivationMode mode = SoftmaxActivationMode::instance) {
  static const char *SoftmaxActivationModeValues[] = {
    "channel",
    "instance"
  };
  return Operator("SoftmaxActivation")
           .SetParam("mode", SoftmaxActivationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Softmax with logit loss.
 *
 *        In the forward pass, the softmax output is returned. Assume the input data has
 *        shape *(n,k)*, then the output will have the same shape as the input, which is
 *
 *        .. math::
 *        out[i,:] = softmax(data[i,:])
 *
 *        for :math:`i=0,...,n-1`, where
 *
 *        .. math::
 *        softmax(x) = \left[..., \frac{exp(x[j])}{exp(x[0])+...+exp(x[k-1])}, ...\right]
 *
 *        For general *N*-D input array with shape :math:`(d_1, ..., d_n)`. Denoted by
 *        :math:`s=d_1d_2...d_n`. The way to compute softmax various:
 *
 *        - ``preserve_shape`` is false (default). Reshape input into a 2-D array with
 *        shape :math:`(d_1, s/d_1)` beforing computing the softmax, and then reshaped
 *        original shape.
 *
 *        - ``preserve_shape`` is true. For all :math:`i_1, ..., i_{n-1}`, compute
 *
 *        .. math::
 *        out[i_1, ..., i_{n-1}, :] = softmax(data[i_1, ..., i_{n-1},:])
 *
 *        - ``multi_output`` is true. For all :math:`i_1, ..., i_{n-1}`, compute
 *
 *        .. math::
 *        out[i_1, :, ..., i_{n-1}] = softmax(data[i_1, :, ..., i_{n-1}])
 *
 *        In the backward pass, the logit loss, also called cross-entroy loss, is
 *        added. The provided label can be a *(N-1)*-D label index array or a *N*-D label
 *        probability array.
 *
 *        Examples with a particular label can be ignored during backward by specifying
 *        ``ignore_label`` (also need ``use_ignore`` to be true).
 *
 *        A scale can be applied to the gradient by ``grad_scale``, which is often used in
 *        mutli-loss object function in which we can given each loss different weight. It
 *        also supports various ways to normalize the gradient by ``normalization``:
 *
 *        - **null**: do nothing
 *        - **batch**: divide by batch size (number of examples)
 *        - **valid**: divide by the number of examples which are not ignored.
 *
 *
 *        Defined in /home/xlidc/mxnet/src/operator/softmax_output.cc:L77
 * \param data Input data.
 * \param label Ground truth label.
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the labels with value equals to ``ignore_label`` will be ignored
 * \param multi_output If set to true, softmax will applied on axis 1
 * \param use_ignore If set to true, the ignore_label value will not contribute to the
 * \param preserve_shape If true, softmax will applied on the last axis
 * \param normalization Normalize the gradient
 * \param out_grad Apply weighting from output gradient
 * \return new symbol
 */
inline Symbol SoftmaxOutput(Symbol data,
                            Symbol label,
                            mx_float grad_scale = 1,
                            mx_float ignore_label = -1,
                            bool multi_output = false,
                            bool use_ignore = false,
                            bool preserve_shape = false,
                            SoftmaxOutputNormalization normalization = SoftmaxOutputNormalization::null,
                            bool out_grad = false) {
  static const char *SoftmaxOutputNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("SoftmaxOutput")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxOutputNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif DEPRECATED: Perform a softmax transformation on input. Please use SoftmaxOutput
 * \param data Input data to softmax.
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the labels with value equals to ``ignore_label`` will be ignored
 * \param multi_output If set to true, softmax will applied on axis 1
 * \param use_ignore If set to true, the ignore_label value will not contribute to the
 * \param preserve_shape If true, softmax will applied on the last axis
 * \param normalization Normalize the gradient
 * \param out_grad Apply weighting from output gradient
 * \return new symbol
 */
inline Symbol Softmax(Symbol data,
                      mx_float grad_scale = 1,
                      mx_float ignore_label = -1,
                      bool multi_output = false,
                      bool use_ignore = false,
                      bool preserve_shape = false,
                      SoftmaxNormalization normalization = SoftmaxNormalization::null,
                      bool out_grad = false) {
  static const char *SoftmaxNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("Softmax")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Apply spatial transformer to input feature map.
 * \param data Input data to the SpatialTransformerOp.
 * \param loc localisation net, the output dim should be 6 when transform_type is affine.
 * \param transform_type transformation type
 * \param sampler_type sampling type
 * \param target_shape output shape(h, w) of spatial transformer: (y, x)
 * \return new symbol
 */
inline Symbol SpatialTransformer(Symbol data,
                                 Symbol loc,
                                 SpatialTransformerTransformType transform_type,
                                 SpatialTransformerSamplerType sampler_type,
                                 Shape target_shape = Shape(0,0)) {
  static const char *SpatialTransformerTransformTypeValues[] = {
    "affine"
  };
  static const char *SpatialTransformerSamplerTypeValues[] = {
    "bilinear"
  };
  return Operator("SpatialTransformer")
           .SetParam("transform_type", SpatialTransformerTransformTypeValues[int(transform_type)])
           .SetParam("sampler_type", SpatialTransformerSamplerTypeValues[int(sampler_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .SetInput("loc", loc)
           .CreateSymbol();
}

/*!
 * \breif Support Vector Machine based transformation on input, backprop L2-SVM
 * \param data Input data to svm.
 * \param label Label data.
 * \param margin Scale the DType(param_.margin) for activation size
 * \param regularization_coefficient Scale the coefficient responsible for balacing
 * \param use_linear If set true, uses L1-SVM objective function. Default uses L2-SVM
 * \return new symbol
 */
inline Symbol SVMOutput(Symbol data,
                        Symbol label,
                        mx_float margin = 1,
                        mx_float regularization_coefficient = 1,
                        bool use_linear = false) {
  return Operator("SVMOutput")
           .SetParam("margin", margin)
           .SetParam("regularization_coefficient", regularization_coefficient)
           .SetParam("use_linear", use_linear)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif generate sampling grid for bilinear sampling.
 * \param data Input data to the GridGeneratorOp.
 * \param transform_type transformation type
 *        if transformation type is affine, data is affine matrix : (batch, 6)
 *        if transformation type is warp, data is optical flow : (batch, 2, h, w)
 * \param target_shape if transformation type is affine, the operator need a target_shape
 *        if transofrmation type is warp, the operator will ignore target_shape
 * \return new symbol
 */
inline Symbol GridGenerator(Symbol data,
                            GridGeneratorTransformType transform_type,
                            Shape target_shape = Shape(0,0)) {
  static const char *GridGeneratorTransformTypeValues[] = {
    "affine",
    "warp"
  };
  return Operator("GridGenerator")
           .SetParam("transform_type", GridGeneratorTransformTypeValues[int(transform_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Choose one element from each line(row for python, column for R/Julia) in lhs
 *        according to index indicated by rhs. This function assume rhs uses 0-based
 * \param lhs Left operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol choose_element_0index(Symbol lhs,
                                    Symbol rhs) {
  return Operator("choose_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Fill one element of each line(row for python, column for R/Julia) in lhs
 *        according to index indicated by rhs and values indicated by mhs. This function
 * \param lhs Left operand to the function.
 * \param mhs Middle operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol fill_element_0index(Symbol lhs,
                                  Symbol mhs,
                                  Symbol rhs) {
  return Operator("fill_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("mhs", mhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

} //namespace cpp
} //namespace mxnet
#endif //ifndef _MXNETOP_H
