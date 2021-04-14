using MxNet.IO;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace MxNet.Image
{
    public class ImageDetIter : ImageIter
    {
        public ImageDetIter(int batch_size, Shape data_shape, string path_imgrec = "", string path_imglist = "",
                            string path_root = "", string path_imgidx = "", bool shuffle = false, int part_index = 0,
                            int num_parts = 1, Augmenter[] aug_list = null, (float[], string)[] imglist = null,
                            string data_name = "data", string label_name = "label", string last_batch_handle = "pad")
            : base(batch_size: batch_size, data_shape: data_shape, path_imgrec: path_imgrec, path_imglist: path_imglist,
                    path_root: path_root, path_imgidx: path_imgidx, shuffle: shuffle, part_index: part_index,
                    num_parts: num_parts, aug_list: aug_list, imglist: imglist, data_name: data_name, label_name: label_name,
                    last_batch_handle: last_batch_handle)
        {
            throw new NotImplementedException();
        }

        private void CheckValidLabel(NDArray label)
        {
            throw new NotImplementedException();
        }

        private Shape EstimateLabelShape()
        {
            throw new NotImplementedException();
        }

        private NDArray ParseLabel(NDArray label)
        {
            throw new NotImplementedException();
        }

        public void Reshape(Shape data_shape = null, Shape label_shape = null)
        {
            throw new NotImplementedException();
        }

        private int Batchify(NDArray batch_data, NDArray batch_label, int start = 0)
        {
            throw new NotImplementedException();
        }

        public override DataBatch Next()
        {
            throw new NotImplementedException();
        }

        public (NDArray, NDArray) AugmentationTransform(NDArray data, NDArray label)
        {
            throw new NotImplementedException();
        }

        public void CheckLabelShape(Shape label_shape)
        {
            throw new NotImplementedException();
        }

        public NDArrayList DrawNext(
                Color color,
                int thickness = 2,
                bool mean = true,
                bool std = true,
                bool clip = true,
                int? waitKey = null,
                string window_name = "draw_next",
                Dictionary<float, string> id2labels = null)
        {
            throw new NotImplementedException();
        }

        public NDArrayList DrawNext(
                Color color,
                int thickness = 2,
                NDArray mean = null,
                NDArray std = null,
                bool clip = true,
                int? waitKey = null,
                string window_name = "draw_next",
                Dictionary<float, string> id2labels = null)
        {
            throw new NotImplementedException();
        }

        public ImageDetIter SyncLabelShape(ImageDetIter it, bool verbose = false)
        {
            throw new NotImplementedException();
        }
    }
}
