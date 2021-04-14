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
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using OpenCvSharp;
using System.Linq;
using System.Runtime.InteropServices;
using NumpyDotNet;
using NumpyDotNet;

namespace MxNet.Recordio
{
    public class IRHeader
    {
        public IRHeader(int flag, NDArray label, int id, int id2)
        {
            Flag = flag;
            Label = label;
            ID = id;
            ID2 = id2;
        }

        public int Flag { get; set; }

        public NDArray Label { get; set; }

        public int ID { get; set; }

        public int ID2 { get; set; }
    }

    public class RecordIO
    {
        public static byte[] Pack(IRHeader header, byte[] s)
        {
            List<byte> bytes = new List<byte>();
            IFormatter formatter = new BinaryFormatter();
            using (MemoryStream stream = new MemoryStream())
            {
                formatter.Serialize(stream, header);
                bytes = stream.ToArray().ToList();
            }

            bytes.AddRange(s);
            return bytes.ToArray();
        }

        public static (IRHeader, byte[]) UnPack(byte[] s)
        {
            int _ir_size = Marshal.SizeOf(typeof(IRHeader));
            var ir_bytes = s.Take(_ir_size).ToArray();
            IRHeader ret = null;
            using (MemoryStream ms = new MemoryStream(ir_bytes))
            {
                IFormatter br = new BinaryFormatter();
                ret = (br.Deserialize(ms) as IRHeader);
            }

            return (ret, s.Skip(_ir_size).ToArray());
        }

        public static (IRHeader, NDArray) UnpackImg(byte[] s, bool iscolor)
        {
            var (header, s_b) = UnPack(s);
            var img = Image.Img.ImDecode(s_b, 1, iscolor);
            return (header, img);
        }

        public static byte[] PackImg(IRHeader header, NDArray img, int quality = 95, string img_fmt = ".jpg")
        {
            int[] encodeParams = null;
            OpenCvSharp.ImageEncodingParam imageEncodingParam = new ImageEncodingParam(ImwriteFlags.JpegQuality, quality);
            if (img_fmt.ToLower() == ".jpg" || img_fmt.ToLower() == ".jpeg")
                encodeParams = new int[] { };

            Mat mat = img;
            Cv2.ImEncode(img_fmt, mat, out var buf, imageEncodingParam);
            return Pack(header, buf);
        }
    }
}