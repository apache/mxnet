using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    internal class ActivationRegistry
    {
        internal static BaseLayer Get(string activationType)
        {
            BaseLayer act = null;

            switch (activationType)
            {
                case ActivationType.ReLU:
                    act = new ReLU();
                    break;
                case ActivationType.Sigmoid:
                    act = new Sigmoid();
                    break;
                case ActivationType.Tanh:
                    act = new Tanh();
                    break;
                case ActivationType.Elu:
                    act = new Elu();
                    break;
                case ActivationType.Exp:
                    act = new Exp();
                    break;
                case ActivationType.HargSigmoid:
                    act = new HardSigmoid();
                    break;
                case ActivationType.LeakyReLU:
                    act = new LeakyReLU();
                    break;
                case ActivationType.PReLU:
                    act = new PReLU();
                    break;
                case ActivationType.RReLU:
                    act = new RReLU();
                    break;
                case ActivationType.SeLU:
                    act = new Selu();
                    break;
                case ActivationType.Softmax:
                    act = new Softmax();
                    break;
                case ActivationType.Softplus:
                    act = new Softplus();
                    break;
                case ActivationType.SoftSign:
                    act = new SoftSign();
                    break;
                default:
                    break;
            }

            return act;
        }
    }
}
