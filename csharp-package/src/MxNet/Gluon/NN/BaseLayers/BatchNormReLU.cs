using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.NN.BaseLayers
{
    public class BatchNormReLU : _BatchNorm
    {
        public BatchNormReLU(int axis = 1, float momentum = 0.9F, float epsilon = 1E-05F, bool center = true, bool scale = true, bool use_global_stats = false, string beta_initializer = "zeros", string gamma_initializer = "ones", string running_mean_initializer = "zeros", string running_variance_initializer = "ones", int in_channels = 0) 
            : base(axis, momentum, epsilon, center, scale, true, use_global_stats, beta_initializer, gamma_initializer, running_mean_initializer, running_variance_initializer, in_channels)
        {
        }
    }
}
