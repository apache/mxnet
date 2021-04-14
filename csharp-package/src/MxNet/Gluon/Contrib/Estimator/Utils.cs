using System;
using MxNet.Gluon.Metrics;

namespace MxNet.Gluon.Contrib
{
    public class Utils
    {
        public static EvalMetric[] CheckMetrics(EvalMetric[] metrics)
        {
            throw new NotImplementedRelease1Exception();
        }

        public static void CheckHandlerMetricRef(IEventHandler handler, EvalMetric[] known_metrics)
        {
            throw new NotImplementedRelease1Exception();
        }

        public static void CheckMetricKnown(IEventHandler handler, EvalMetric metric, EvalMetric[] known_metrics)
        {
            throw new NotImplementedRelease1Exception();
        }

        public static EvalMetric SuggestMetricForLoss(Losses.Loss loss)
        {
            throw new NotImplementedRelease1Exception();
        }
    }
}
