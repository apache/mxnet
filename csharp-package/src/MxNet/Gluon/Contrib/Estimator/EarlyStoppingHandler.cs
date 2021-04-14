using System;
namespace MxNet.Gluon.Contrib
{
    public class EarlyStoppingHandler : IEventHandler
    {
        public EarlyStoppingHandler(Monitor monitor,
                 float min_delta= 0,
                 int patience= 0,
                 string mode= "auto",
                 float? baseline= null)
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
