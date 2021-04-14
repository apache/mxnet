using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Contrib
{
    public class StoppingHandler : IEventHandler
    {
        public StoppingHandler(int? max_epoch= null, int? max_batch= null)
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
        }
    }
}
