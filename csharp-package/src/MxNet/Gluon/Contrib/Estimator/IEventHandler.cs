using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Contrib
{
    public interface IEventHandler
    {
        void TrainBegin(Estimator estimator);

        void TrainEnd(Estimator estimator);

        void EpochBegin(Estimator estimator);

        bool EpochEnd(Estimator estimator);

        void BatchBegin(Estimator estimator);

        bool BatchEnd(Estimator estimator);
    }
}
