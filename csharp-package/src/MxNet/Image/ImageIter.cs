/*****************************************************************************
   Copyright 2018 The MxNet.Sharp Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using MxNet.IO;
using MxNet.Recordio;
using OpenCvSharp;

namespace MxNet.Image
{
    public class ImageIter : DataIter
    {
        public bool _allow_read;

        public NDArray _cache_data;

        public int? _cache_idx;

        public NDArray _cache_label;

        public Augmenter[] auglist;

        public int batch_size;

        public int cur;

        public Shape data_shape;

        public List<string> imgidx;

        public Dictionary<int, (NDArray, string)> imglist;

        public MXIndexedRecordIO imgrec;

        public int label_width;

        public string last_batch_handle;

        public int num_image;

        public string path_root;

        public DataDesc[] provide_data;

        public DataDesc[] provide_label;

        public List<string> seq;

        public bool shuffle;

        public ImageIter(int batch_size, Shape data_shape, int label_width = 1,
            string path_imgrec = null, string path_imglist = null, string path_root = null,
            string path_imgidx = null, bool shuffle = false, int part_index = 0, int num_parts = 1,
            Augmenter[] aug_list = null, (float[], string)[] imglist = null,
            string data_name = "data", string label_name = "softmax_label", DType dtype = null,
            string last_batch_handle = "pad") : base(batch_size)
        {
            Debug.Assert(path_imgrec != null || path_imglist != null);
            Debug.Assert(new List<string> {
                "int32",
                "float32",
                "int64",
                "float64"
            }.Contains(dtype), dtype + " label not supported");
            int num_threads = 1;
            if(!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MXNET_CPU_WORKER_NTHREADS")))
                num_threads = Convert.ToInt32(Environment.GetEnvironmentVariable("MXNET_CPU_WORKER_NTHREADS"));

            Logger.Info("Using %s threads for decoding..." + num_threads.ToString());
            Logger.Info("Set enviroment variable MXNET_CPU_WORKER_NTHREADS to a larger number to use more threads.");
            var class_name = this.GetType().Name;
            if (!string.IsNullOrWhiteSpace(path_imgrec))
            {
                Logger.Info($"{class_name}: loading recordio {path_imgrec}...");
                if (!string.IsNullOrWhiteSpace(path_imgidx))
                {
                    this.imgrec = new MXIndexedRecordIO(path_imgidx, path_imgrec, "r");
                    this.imgidx = this.imgrec.Keys;
                }
                else
                {
                    this.imgrec = new MXIndexedRecordIO("", path_imgrec, "r");
                    this.imgidx = null;
                }
            }
            else
            {
                this.imgrec = null;
            }
            var imgkeys = new List<string>();
            if (path_imglist != null)
            {
                Logger.Info($"{class_name}: loading image list {path_imglist}...");
                var lines = File.ReadAllLines(path_imglist);
                this.imglist = new Dictionary<int, (NDArray, string)>();
                
                foreach (var line in lines)
                {
                    var splitLines = line.Trim().Split('\t');
                    var label = nd.Array(splitLines.Skip(1).Take(splitLines.Length-2).Select(x=>Convert.ToSingle(x)).ToArray()).AsType(dtype);
                    var key = splitLines[0];
                    this.imglist[Convert.ToInt32(key)] = (label, splitLines.Last());
                    imgkeys.Add(key);
                }
            }
            else if (imglist != null)
            {
                Logger.Info(@"{class_name}: loading image list...");
                this.imglist = new Dictionary<int, (NDArray, string)>();
                var index = 1;
                foreach (var (i, s) in imglist)
                {
                    var key = index.ToString();
                    index += 1;
                    var label = nd.Array(i).AsType(dtype);
                    this.imglist[index] = (label, s);
                    imgkeys.Add(key);
                }
            }
            else
            {
                this.imglist = null;
            }

            this.path_root = path_root;
            this.CheckDataShape(data_shape);
            var pshape = data_shape.Data.ToList();
            pshape.Insert(0, batch_size);
            this.provide_data = new DataDesc[] { new DataDesc(data_name, new Shape(pshape)) };

            if (label_width > 1)
            {
                this.provide_label = new DataDesc[] { new DataDesc(label_name, new Shape(batch_size, label_width)) };
            }
            else
            {
                this.provide_label = new DataDesc[] { new DataDesc(label_name, new Shape(batch_size)) };
            }

            this.batch_size = batch_size;
            this.data_shape = data_shape;
            this.label_width = label_width;
            this.shuffle = shuffle;
            if (this.imgrec == null)
            {
                this.seq = imgkeys;
            }
            else if (shuffle || num_parts > 1 || path_imgidx != null)
            {
                Debug.Assert(this.imgidx != null);
                this.seq = this.imgidx;
            }
            else
            {
                this.seq = null;
            }

            if (num_parts > 1)
            {
                Debug.Assert(part_index < num_parts);
                var N = this.seq.Count;
                var C = N / num_parts;
                this.seq = this.seq.Skip((part_index * C)).Take(((part_index + 1) * C)).ToList();
            }
            if (aug_list == null)
            {
                this.auglist = Img.CreateAugmenter(data_shape);
            }
            else
            {
                this.auglist = aug_list;
            }
            this.cur = 0;
            this._allow_read = true;
            this.last_batch_handle = last_batch_handle;
            this.num_image = this.seq != null ? this.seq.Count : 0;
            this._cache_data = null;
            this._cache_label = null;
            this._cache_idx = null;
            this.Reset();
        }

        public override NDArrayList GetData()
        {
            throw new NotImplementedException();
        }

        public override int[] GetIndex()
        {
            throw new NotImplementedException();
        }

        public override NDArrayList GetLabel()
        {
            throw new NotImplementedException();
        }

        public override int GetPad()
        {
            throw new NotImplementedException();
        }

        public override bool IterNext()
        {
            throw new NotImplementedException();
        }

        public override bool End()
        {
            throw new NotImplementedException();
        }

        public override DataBatch Next()
        {
            int i;
            NDArray batch_label;
            NDArray batch_data;
            var batch_size = this.batch_size;
            var (c, h, w) = this.data_shape;
            // if last batch data is rolled over
            if (this._cache_data != null)
            {
                // check both the data and label have values
                Debug.Assert(this._cache_label != null, "_cache_label didn't have values");
                Debug.Assert(this._cache_idx != null, "_cache_idx didn't have values");
                batch_data = this._cache_data;
                batch_label = this._cache_label;
                i = this._cache_idx.Value;
                // clear the cache data
            }
            else
            {
                batch_data = nd.Zeros(new Shape(batch_size, c, h, w));
                batch_label = nd.Empty(this.provide_label[0].Shape);
                i = this.Batchify(batch_data, batch_label);
            }
            // calculate the padding
            var pad = batch_size - i;
            // handle padding for the last batch
            if (pad != 0)
            {
                if (this.last_batch_handle == "discard")
                {
                    // pylint: disable=no-else-raise
                    throw new Exception("Stop Iteraion");
                }
                else if (this.last_batch_handle == "roll_over" && this._cache_data == null)
                {
                    // if the option is 'roll_over', throw StopIteration and cache the data
                    this._cache_data = batch_data;
                    this._cache_label = batch_label;
                    this._cache_idx = i;
                    throw new StopIteration();
                }
                else
                {
                    var _ = this.Batchify(batch_data, batch_label, i);
                    if (this.last_batch_handle == "pad")
                    {
                        this._allow_read = false;
                    }
                    else
                    {
                        this._cache_data = null;
                        this._cache_label = null;
                        this._cache_idx = null;
                    }
                }
            }
            return new DataBatch(batch_data, batch_label, pad: pad);
        }

        public (NDArray, byte[]) NextSample()
        {
            byte[] img = null;
            IRHeader header = null;
            byte[] s;
            if (!_allow_read)
            {
                throw new Exception("Stop iteration");
            }
            if (this.seq != null)
            {
                int idx = 0;
                if (this.cur < this.num_image)
                {
                    idx = Convert.ToInt32(this.seq[this.cur]);
                }
                else
                {
                    if (this.last_batch_handle != "discard")
                    {
                        this.cur = 0;
                    }
                    throw new Exception("Stop iteration");
                }
                this.cur += 1;
                if (this.imgrec != null)
                {
                    s = this.imgrec.ReadIdx(Convert.ToInt32(idx));
                    (header, img) = RecordIO.UnPack(s);
                    if (this.imglist == null)
                    {
                        return (header.Label, img);
                    }
                    else
                    {
                        return (this.imglist[idx].Item1, img);
                    }
                }
                else
                {
                    var _tup_2 = this.imglist[idx];
                    var label = _tup_2.Item1;
                    var fname = _tup_2.Item2;
                    return (label, this.ReadImage(fname));
                }
            }
            else
            {
                s = this.imgrec.Read();
                if (s == null)
                {
                    if (this.last_batch_handle != "discard")
                    {
                        this.imgrec.Reset();
                    }

                    throw new Exception("Stop Iteration");
                }

                (header, img) = RecordIO.UnPack(s);
                return (header.Label, img);
            }
        }

        public override void Reset()
        {
            if (this.seq != null && this.shuffle)
            {
                this.seq.Shuffle();
            }
            if (this.last_batch_handle != "roll_over" || this._cache_data == null)
            {
                if (this.imgrec != null)
                {
                    this.imgrec.Reset();
                }
                this.cur = 0;
                if (!_allow_read)
                {
                    this._allow_read = true;
                }
            }
        }

        public void HardReset()
        {
            if (this.seq != null && this.shuffle)
            {
                this.seq.Shuffle();
            }

            if (this.imgrec != null)
            {
                this.imgrec.Reset();
            }
            this.cur = 0;
            this._allow_read = true;
            this._cache_data = null;
            this._cache_label = null;
            this._cache_idx = null;
        }

        private int Batchify(NDArrayList batch_data, NDArrayList batch_label, int start = 0)
        {
            var i = start;
            var batch_size = this.batch_size;
            try
            {
                while (i < batch_size)
                {
                    var _tup_1 = this.NextSample();
                    var label = _tup_1.Item1;
                    var s = _tup_1.Item2;
                    var data = this.ImDecode(s);
                    try
                    {
                        this.CheckValidImage(data);
                    }
                    catch (Exception e)
                    {
                        Logger.Info("Invalid image, skipping: " + e.Message);
                        continue;
                    }

                    data = this.AugmentationTransform(data);
                    Debug.Assert(i < batch_size, "Batch size must be multiples of augmenter output length");
                    batch_data[i] = this.PostProcessData(data);
                    batch_label[i] = label;
                    i += 1;
                }
            }
            catch (Exception)
            {
                throw new Exception("Stop iteration");
            }

            return i;
        }

        public void CheckDataShape(Shape data_shape)
        {
            if (!(data_shape.Dimension == 3))
            {
                throw new Exception("data_shape should have length 3, with dimensions CxHxW");
            }
            if (!(data_shape[0] == 3))
            {
                throw new Exception("This iterator expects inputs to have 3 channels.");
            }
        }

        public void CheckValidImage(NDArray data)
        {
            if (data.Shape.Dimension== 0)
            {
                throw new Exception("Data shape is wrong");
            }
        }

        public NDArray ImDecode(byte[] s)
        {
            NDArray ret = Cv2.ImDecode(s, ImreadModes.Color);
            return ret;
        }

        public byte[] ReadImage(string fname)
        {
            var mat = Cv2.ImRead(fname);
            int size = mat.Channels() * mat.Width * mat.Height;
            byte[] bytes = new byte[size];
            unsafe
            {
                Buffer.MemoryCopy(mat.Data.ToPointer(), bytes.GetMemPtr().ToPointer(), bytes.Length, bytes.Length);
            }

            return bytes;
        }

        public NDArray AugmentationTransform(NDArray data)
        {
            foreach (var aug in this.auglist)
            {
                data = aug.Call(data);
            }

            return data;
        }

        public NDArray PostProcessData(NDArray datum)
        {
            return nd.Transpose(datum, axes: new Shape(2, 0, 1));
        }
    }
}