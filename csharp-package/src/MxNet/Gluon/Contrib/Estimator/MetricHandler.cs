using MxNet.Gluon.Metrics;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Contrib
{
    public class MetricHandler : IEventHandler
    {
        public MetricHandler(EvalMetric[] metrics, int priority = 1000)
        {
            throw new NotImplementedRelease1Exception();
        }

        public void BatchBegin(Estimator estimator)
        {
            
        }

        public bool BatchEnd(Estimator estimator)
        {
            throw new NotImplementedRelease1Exception();
        }

        public void EpochBegin(Estimator estimator)
        {
            throw new NotImplementedRelease1Exception();
        }

        public bool EpochEnd(Estimator estimator)
        {
            return true;
        }

        public void TrainBegin(Estimator estimator)
        {
            
        }

        public void TrainEnd(Estimator estimator)
        {
            
        }
    }
}
