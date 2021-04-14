using MxNet.Gluon.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Contrib
{
    public class ValidationHandler : IEventHandler
    {
        public ValidationHandler(DataLoader val_data,
                 Action<NDArrayList, int, IEventHandler[]> eval_fn,
                 int epoch_period = 1,
                 int? batch_period = null,
                 int priority = -1000,
                 IEventHandler[] event_handlers = null)
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
