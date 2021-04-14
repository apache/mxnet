using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN
{
    public class LossRegistry
    {
        public static Symbol Get(LossType lossType, Symbol preds, Symbol labels)
        {
            switch (lossType)
            {
                case LossType.MeanSquaredError:
                    return MeanSquaredError(preds, labels);
                case LossType.MeanAbsoluteError:
                    return MeanAbsoluteError(preds, labels);
                case LossType.MeanAbsolutePercentageError:
                    return MeanAbsolutePercentageError(preds, labels);
                case LossType.MeanAbsoluteLogError:
                    return MeanAbsoluteLogError(preds, labels);
                case LossType.SquaredHinge:
                    return SquaredHinge(preds, labels);
                case LossType.Hinge:
                    return Hinge(preds, labels);
                case LossType.SigmoidBinaryCrossEntropy:
                    return SigmoidBinaryCrossEntropy(preds, labels);
                case LossType.SoftmaxCategorialCrossEntropy:
                    return SoftmaxCategorialCrossEntropy(preds, labels);
                case LossType.CTC:
                    return CTC(preds, labels);
                case LossType.KullbackLeiblerDivergence:
                    return KullbackLeiblerDivergence(preds, labels);
                case LossType.Poisson:
                    return Poisson(preds, labels);
                default:
                    return null;
            }
        }

        private static Symbol MeanSquaredError(Symbol preds, Symbol labels)
        {
            return sym.LinearRegressionOutput(preds, labels);
        }

        private static Symbol MeanAbsoluteError(Symbol preds, Symbol labels)
        {
            return sym.MAERegressionOutput(preds, labels);
        }

        private static Symbol MeanAbsolutePercentageError(Symbol preds, Symbol labels)
        {
            Symbol loss = sym.Mean(sym.Abs(labels - preds) / sym.Clip(sym.Abs(labels), float.Epsilon, 0));
            return new Operator("MakeLoss").SetInput("data", loss).CreateSymbol("MeanAbsolutePercentageError");
        }

        private static Symbol MeanAbsoluteLogError(Symbol preds, Symbol labels)
        {
            Symbol first_log = sym.Log(sym.Clip(preds, float.Epsilon, 0) + 1);
            Symbol second_log = sym.Log(sym.Clip(labels, float.Epsilon, 0) + 1);
            Symbol loss = sym.Mean(sym.Square(first_log - second_log));
            return new Operator("MakeLoss").SetInput("data", loss).CreateSymbol("MeanAbsoluteLogError");
        }

        private static Symbol SquaredHinge(Symbol preds, Symbol labels)
        {
            Symbol loss = sym.Mean(sym.Square(sym.MaximumScalar(1 - (labels * preds), 0)));
            return new Operator("MakeLoss").SetInput("data", loss).CreateSymbol("SquaredHinge");
        }

        private static Symbol Hinge(Symbol preds, Symbol labels)
        {
            Symbol loss = sym.Mean(sym.MaximumScalar(1 - (labels * preds), 0));
            return new Operator("MakeLoss").SetInput("data", loss).CreateSymbol("Hinge");
        }

        private static Symbol SigmoidBinaryCrossEntropy(Symbol preds, Symbol labels)
        {
            return sym.LogisticRegressionOutput(preds, labels);
        }

        private static Symbol SoftmaxCategorialCrossEntropy(Symbol preds, Symbol labels)
        {
            return sym.SoftmaxOutput(preds, labels);
        }

        private static Symbol CTC(Symbol preds, Symbol labels)
        {
            return sym.CTCLoss(preds, labels, null, null);
        }

        private static Symbol KullbackLeiblerDivergence(Symbol preds, Symbol labels)
        {
            Symbol y_true = sym.Clip(labels, float.Epsilon, 1);
            Symbol y_pred = sym.Clip(preds, float.Epsilon, 1);
            Symbol loss = sym.Sum(y_true * sym.Log(y_true / y_pred));
            return new Operator("MakeLoss").SetInput("data", loss).CreateSymbol("KullbackLeiblerDivergence");
        }

        private static Symbol Poisson(Symbol preds, Symbol labels)
        {
            Symbol loss = sym.Mean(preds - labels * sym.Log(preds + float.Epsilon));
            return new Operator("MakeLoss").SetInput("data", loss).CreateSymbol("Poisson");
        }
    }
}
