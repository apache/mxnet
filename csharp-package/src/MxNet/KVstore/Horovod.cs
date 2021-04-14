using MxNet.Optimizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.KVstore
{
    public class Horovod : KVStoreBase
    {
        public override int Rank => throw new NotImplementedException();

        public override string Type => throw new NotImplementedException();

        public override int NumWorkers => throw new NotImplementedException();

        public override void Broadcast(string key, NDArrayList value, NDArrayList @out, int priority = 0)
        {
            throw new NotImplementedException();
        }

        public override void Broadcast(int key, NDArrayList value, NDArrayList @out, int priority = 0)
        {
            throw new NotImplementedException();
        }

        public override bool IsCapable(string capability)
        {
            throw new NotImplementedException();
        }

        public override void LoadOptimizerStates(string fname)
        {
            throw new NotImplementedException();
        }

        public override void PushPull(string key, NDArrayList value, NDArrayList @out, int priority = 0)
        {
            throw new NotImplementedException();
        }

        public override void PushPull(int key, NDArrayList value, NDArrayList @out, int priority = 0)
        {
            throw new NotImplementedException();
        }

        public override void SaveOptimizerStates(string fname, bool dump_optimizer = false)
        {
            throw new NotImplementedException();
        }

        public override void SetOptimizer(Optimizer optimizer)
        {
            throw new NotImplementedException();
        }
    }
}
