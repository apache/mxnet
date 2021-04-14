using MxNet;
using MxNet.Gluon;
using MxNet.Gluon.ModelZoo.Vision;
using MxNet.GluonCV.Data.Transforms.Presets;
using MxNet.GluonCV.ModelZoo.Yolo;
using MxNet.GluonCV.Utils;
using MxNet.Image;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GluonCVExamples
{
    public class DarknetExamples
    {
        public static void RunDetection()
        {
            var net = YOLOV3.YOLO3_Darknet53_VOC(pretrained: true);
            var im_fname = Utils.Download("https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg", "objdet.jpg");
            var (x, img) = Yolo.LoadTest(im_fname, @short: 512);
            Img.ImShow(x);
            Console.WriteLine("Shape of pre-processed image:" + x.Shape);
            var (class_IDs, scores, bounding_boxs) = net.Call(x.AsType(DType.Float32));
            img = Viz.PlotBBox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names: net.Classes);
            Img.ImShow(img);
        }

        public static void RunClassification()
        {
            var net = DarknetV3.Darknet53(pretrained: true);
            var image = Img.ImRead("objdet.jpg");
            NDArray transformed_img = Imagenet.TransformEval(image, 512, 512);
            var pred = net.Call(transformed_img);
            NDArray prob = nd.Topk(nd.Softmax(pred), k: 5);
            var label_index = prob.ArrayData.OfType<float>().ToList();
            var imagenet_labels = TestUtils.GetImagenetLabels();
            foreach (int i in label_index)
            {
                Console.WriteLine(imagenet_labels[i]);
            }
        }
    }
}
