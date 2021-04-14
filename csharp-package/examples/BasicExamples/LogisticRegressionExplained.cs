using MxNet;
using MxNet.Gluon;
using MxNet.Gluon.Data;
using MxNet.Gluon.Losses;
using MxNet.Gluon.NN;
using MxNet.Initializers;
using MxNet.Gluon.Metrics;
using MxNet.Optimizers;
using System;
using System.Linq;

namespace BasicExamples
{
    public class LogisticRegressionExplained
    {
        private static HybridSequential net = null;
        private static SigmoidBinaryCrossEntropyLoss loss = null;
        private static Trainer trainer = null;
        private static Accuracy accuracy = null;
        private static F1 f1 = null;
        private static DataLoader train_dataloader = null;
        private static DataLoader val_dataloader = null;

        private static int batch_size = 10;
        
        public static void Run()
        {
            //Logistic Regression is one of the first models newcomers to Deep Learning are implementing. 
            //The focus of this tutorial is to show how to do logistic regression using Gluon API.

            var ctx = mx.Cpu();
            int train_data_size = 1000;
            int val_data_size = 100;

            var (train_x, train_ground_truth_class) = GetRandomData(train_data_size, ctx);
            var train_dataset = new ArrayDataset((train_x, train_ground_truth_class));
            train_dataloader = new DataLoader(train_dataset, batch_size: batch_size, shuffle: true);

            var (val_x, val_ground_truth_class) = GetRandomData(val_data_size, ctx);
            var val_dataset = new ArrayDataset((val_x, val_ground_truth_class));
            val_dataloader = new DataLoader(val_dataset, batch_size: batch_size, shuffle: true);

            net = new HybridSequential();
            net.Add(new Dense(units: 10, activation: ActivationType.Relu));
            net.Add(new Dense(units: 10, activation: ActivationType.Relu));
            net.Add(new Dense(units: 10, activation: ActivationType.Relu));
            net.Add(new Dense(units: 1));

            net.Initialize(new Xavier());
            loss = new SigmoidBinaryCrossEntropyLoss();
            trainer = new Trainer(net.CollectParams(), new SGD(learning_rate: 0.1f));

            accuracy = new Accuracy();
            f1 = new F1();

            int epochs = 10;
            float threshold = 0.5f;

            foreach (var e in Enumerable.Range(0, epochs))
            {
                var avg_train_loss = TrainModel() / train_data_size;
                var avg_val_loss = ValidateModel(threshold) / val_data_size;
                Console.WriteLine($"Epoch: {e}, Training loss: {avg_train_loss}, Validation loss: {avg_val_loss}, Validation accuracy: {accuracy.Get().Item2}, F1 score: {f1.Get().Item2}");

                accuracy.Reset();
            }
        }

        private static float TrainModel()
        {
            float cumulative_train_loss = 0;
            foreach (var (data, label) in train_dataloader)
            {
                NDArray loss_result = null;
                using (var ag = Autograd.Record())
                {
                    var output = net.Call(data);
                    loss_result = loss.Call(output, label);
                    loss_result.Backward();
                }
                
                trainer.Step(batch_size);
                cumulative_train_loss += nd.Sum(loss_result).AsScalar<float>();
            }

            return cumulative_train_loss;
        }

        private static float ValidateModel(float threshold)
        {
            float cumulative_val_loss = 0;
            foreach (var (val_data, val_ground_truth_class) in val_dataloader)
            {
                var output = net.Call(val_data);
                NDArray loss_result = loss.Call(output, val_ground_truth_class);
                
                cumulative_val_loss += nd.Sum(loss_result).AsScalar<float>();

                NDArray prediction = net.Call(val_data);
                prediction = prediction.Sigmoid();
                var predicted_classes = nd.Ceil(prediction - threshold);
                accuracy.Update(val_ground_truth_class, predicted_classes.Reshape(-1));
                prediction = prediction.Reshape(-1);
                var probabilities = nd.Stack(new NDArrayList(1 - prediction, prediction), 2, axis: 1);
                f1.Update(val_ground_truth_class, probabilities);
            }

            return cumulative_val_loss;
        }

        /// <summary>
        /// In this tutorial we will use fake dataset, which contains 10 features drawn from a normal distribution with mean equals to 0 and
        /// standard deviation equals to 1, and a class label, which can be either 0 or 1. The size of the dataset is an arbitrary value. 
        /// The function below helps us to generate a dataset. Class label y is generated via a non-random logic, 
        /// so the network would have a pattern to look for. Boundary of 3 is selected to make sure that number 
        /// of positive examples smaller than negative, but not too small
        /// </summary>
        /// <param name="size"></param>
        /// <param name="ctx"></param>
        /// <returns></returns>
        private static (NDArray, NDArray) GetRandomData(int size, Context ctx)
        {
            var x = nd.Random.Normal(0, 1, shape: new Shape(size, 10), ctx: ctx);
            var y = x.Sum(axis: 1) > 3;
            return (x, y);
        }
    }
}
