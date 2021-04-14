using System;
namespace MxNet.Gluon.Contrib
{
    public class GradientUpdateHandler : IEventHandler
    {
        public GradientUpdateHandler(int priority = -2000)
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
