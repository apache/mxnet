using MxNet;
using MxNet.Gluon.ModelZoo.Vision;
using MxNet.Image;
using MxNet.ND.Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ImageClassification
{
    public class ResnetGluon
    {
        public static void Run()
        {
            var net = ResNet.ResNet18_v2(true);
            var image = Img.ImRead("goldfish.jpg");
            image = Img.ResizeShort(image, 227);
            image = image.AsType(DType.Float32) / 255;
            var normalized = Img.ColorNormalize(image, new NDArray(new[] { 0.485f, 0.456f, 0.406f }),
                new NDArray(new[] { 0.229f, 0.224f, 0.225f }));
            normalized = normalized.transpose(new Shape(2, 0, 1));
            normalized = normalized.expand_dims(axis: 0);
            var pred = net.Call(normalized);
            NDArray prob = npx.topk(npx.softmax(pred), k: 5);
            var label_index = prob.ArrayData.OfType<float>().ToList();
            var imagenet_labels = TestUtils.GetImagenetLabels();
            foreach (int i in label_index)
            {
                Console.WriteLine(imagenet_labels[i]);
            }
        }
    }
}
