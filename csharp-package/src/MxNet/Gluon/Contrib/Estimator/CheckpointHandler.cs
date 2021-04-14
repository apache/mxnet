using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Contrib
{
    public class CheckpointHandler : IEventHandler
    {
        public CheckpointHandler(string model_dir,
                 string model_prefix= "model",
                 Monitor monitor= null,
                 int verbose= 0,
                 bool save_best= false,
                 string mode= "auto",
                 int epoch_period= 1,
                 int? batch_period= null,
                 int max_checkpoints= 5,
                 bool resume_from_checkpoint= false)
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
