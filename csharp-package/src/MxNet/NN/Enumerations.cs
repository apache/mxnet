using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN
{
    public class ActivationType
    {
        public const string Linear = "linear";
        public const string ReLU = "relu";
        public const string Sigmoid = "sigmoid";
        public const string Tanh = "tanh";
        public const string Elu = "elu";
        public const string Exp = "exp";
        public const string HargSigmoid = "hard_sigmoid";
        public const string LeakyReLU = "leaky_relu";
        public const string PReLU = "p_relu";
        public const string RReLU = "r_relu";
        public const string SeLU = "selu";
        public const string Softmax = "softmax";
        public const string Softplus = "softplus";
        public const string SoftSign = "softsign";
    }

    public enum OptimizerType
    {
        SGD,
        Signum,
        RMSprop,
        Adagrad,
        Adadelta,
        Adam
    }

    public enum LossType
    {
        MeanSquaredError,
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        MeanAbsoluteLogError,
        SquaredHinge,
        Hinge,
        SigmoidBinaryCrossEntropy,
        SoftmaxCategorialCrossEntropy,
        CTC,
        KullbackLeiblerDivergence,
        Poisson
    }
}
