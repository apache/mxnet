using MxNet.Gluon.Data;
using MxNet.Gluon.Losses;
using MxNet.Gluon.Metrics;
using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Contrib
{
    public class Estimator
    {
        public EvalMetric[] TrainMetrics { get; set; }

        public EvalMetric[] ValMetrics { get; set; }

        public Estimator(Block net, Losses.Loss loss, EvalMetric[] train_metrics = null, EvalMetric[] val_metrics = null, Trainer trainer = null,
                           Context context= null, Block val_net= null, Losses.Loss val_loss= null, BatchProcessor batch_processor= null)
        {
            throw new NotImplementedRelease1Exception();
        }

        private Losses.Loss CheckLoss(Losses.Loss loss)
        {
            throw new NotImplementedRelease1Exception();
        }

        private Context CheckContext(Context context)
        {
            throw new NotImplementedRelease1Exception();
        }

        private BatchProcessor CheckBatchProcessor(BatchProcessor batch_processor)
        {
            throw new NotImplementedRelease1Exception();
        }

        private void Initialize(Initializers.Initializer initializer)
        {
            throw new NotImplementedRelease1Exception();
        }

        private Trainer CheckTrainer(Trainer trainer)
        {
            throw new NotImplementedRelease1Exception();
        }

        private bool IsInitialized()
        {
            throw new NotImplementedRelease1Exception();
        }

        private (ndarray, ndarray) GetDataAndLabel(NDArrayList batch, Context ctx, int batch_axis= 0)
        {
            throw new NotImplementedRelease1Exception();
        }

        private void AddDefaultTrainingMetrics()
        {
            throw new NotImplementedRelease1Exception();
        }

        private void AddValidationMetrics()
        {
            throw new NotImplementedRelease1Exception();
        }

        public void Evaluate(DataLoader val_data, int batch_axis= 0, IEventHandler[] event_handlers= null)
        {
            throw new NotImplementedRelease1Exception();
        }

        public void Fit(DataLoader train_data, DataLoader val_data, int? epochs = null, IEventHandler[] event_handlers = null, int? batches = null, int batch_axis = 0)
        {
            throw new NotImplementedRelease1Exception();
        }

        private IEventHandler[] PrepareDefaultHandlers(DataLoader val_data, IEventHandler[]  event_handlers)
        {
            throw new NotImplementedRelease1Exception();
        }

        private IEventHandler[] PrepareDefaultValidationHandlers(DataLoader val_data, IEventHandler[] event_handlers)
        {
            throw new NotImplementedRelease1Exception();
        }

        private IEventHandler[] CategorizeHandlers(DataLoader val_data, IEventHandler[] event_handlers)
        {
            throw new NotImplementedRelease1Exception();
        }
    }
}
