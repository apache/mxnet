using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Contrib
{
    public class BatchProcessor
    {
        public (ndarray, ndarray) GetDataAndLabel(int batch, Context ctx, int batch_axis = 0)
        {
            throw new NotImplementedRelease1Exception();
        }

        public (ndarray, ndarray, ndarray, ndarray) FitBatch(Estimator estimator, (ndarray, ndarray) train_batch, int batch_axis = 0)
        {
            throw new NotImplementedRelease1Exception();
        }

        public (ndarray, ndarray, ndarray, ndarray) EvaluateBatch(Estimator estimator, (ndarray, ndarray) val_batch, int batch_axis = 0)
        {
            throw new NotImplementedRelease1Exception();
        }
    }
}
