using MxNet.Gluon.Metrics;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Contrib
{
    public class LoggingHandler : IEventHandler
    {
        public LoggingHandler(string log_interval= "epoch",
                 EvalMetric[] metrics= null,
                 float priority= float.PositiveInfinity)
        {
            throw new NotImplementedRelease1Exception();
        }

        public void BatchBegin(Estimator estimator)
        {
            throw new NotImplementedRelease1Exception();
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
            throw new NotImplementedRelease1Exception();
        }

        public void TrainBegin(Estimator estimator)
        {
            throw new NotImplementedRelease1Exception();
        }

        public void TrainEnd(Estimator estimator)
        {
            throw new NotImplementedRelease1Exception();
        }
    }
}
