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
using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

namespace MxNet.Gluon
{
    public class Utils
    {
        private static readonly SemaphoreSlim _semaphore = new SemaphoreSlim(0);

        public static NDArrayList SplitData(ndarray data, int num_slice, int batch_axis = 0, bool even_split = true)
        {
            var size = data.shape[batch_axis];
            if (even_split && size % num_slice != 0)
                throw new ArgumentException(string.Format(
                    "data with shape {0} cannot be evenly split into {1} slices along axis {2}. " +
                    "Use a batch size that's multiple of {3} or set even_split=False to allow " +
                    "uneven partitioning of data.", data.shape, num_slice, batch_axis, num_slice));

            var step = (int) Math.Truncate((double) size / num_slice);

            if (!even_split && size < num_slice)
            {
                step = 1;
                num_slice = size;
            }

            var slices = new NDArrayList();
            if (batch_axis == 0)
            {
                for (var i = 0; i < num_slice; i++)
                    if (i < num_slice)
                        slices.Add(data[string.Format("{0}:{1}", i * step, (i + 1) * step)]);
            }
            else if (even_split)
            {
                slices.Add(nd.Split(data, num_slice, batch_axis));
            }
            else
            {
                for (var i = 0; i < num_slice; i++)
                    if (i < num_slice - 1)
                        slices.Add(data[string.Format("{0}:{1}", i * step, (i + 1) * step)]);
                    else
                        slices.Add(nd.SliceAxis(data, batch_axis, i * step, (i + 1) * step));
            }

            return slices.ToArray();
        }

        public static NDArrayList SplitAndLoad(ndarray data, Context[] ctx_list, int batch_axis = 0,
            bool even_split = true)
        {
            if (ctx_list.Length == 1)
                return data.AsInContext(ctx_list[0]);

            var slices = SplitData(data, ctx_list.Length, batch_axis, even_split);

            var result = new NDArrayList();
            result = slices.Zip(ctx_list, (i, ctx) => { return i.AsInContext(ctx); }).ToList();

            return result.ToArray();
        }

        public static ndarray ClipGlobalNorm(NDArrayList arrays, float max_norm, bool check_isfinite = true)
        {
            Func<NDArrayList, Dictionary<Context, NDArrayList>> group_by_ctx = arr_list =>
            {
                Dictionary<Context, NDArrayList> groups = new Dictionary<Context, NDArrayList>();
                foreach (var arr in arr_list)
                {
                    if(groups.ContainsKey(arr.ctx))
                    {
                        groups[arr.ctx].Add(arr);
                    }
                    else
                    {
                        groups.Add(arr.ctx, arr);
                    }
                }

                return groups;
            };

            if (arrays.Length == 0)
                throw new ArgumentException("arrays.Length == 0");

            var arrays_groups = group_by_ctx(arrays);
            var all_ctx_sum = new NDArrayList();
            var ctx = arrays[0].ctx;
            foreach (var group in arrays_groups)
            {
                var sum_sq = nd.MultiSumSq(group.Value, num_arrays: group.Value.Length);
                sum_sq = nd.AddN(sum_sq);
                all_ctx_sum.Add(sum_sq[0].AsInContext(ctx));
            }

            var total_norm = nd.AddN(arrays.Select(x => x.AsInContext(ctx)).ToArray());
            total_norm = total_norm.Sqrt();
            if (check_isfinite)
                if (float.IsInfinity(total_norm.AsScalar<float>()))
                    Logger.Warning("nan or inf is detected. " +
                                   "Clipping results will be undefined.");

            var scale = max_norm / (total_norm + 1e-8f);
            scale = np.min(np.concatenate(new NDArrayList(scale, np.ones(new Shape(1), ctx: ctx)), 0));
            for (var i = 0; i < arrays.Length; i++) arrays[i] *= (ndarray)scale.AsInContext(arrays[i].ctx);

            return total_norm;
        }

        public static string Indent(string s_, int numSpaces)
        {
            var s = s_.Split('\n');
            if (s.Length == 1)
                return s_;

            var result = new StringBuilder(s[0]);
            foreach (var item in s.Skip(1))
            {
                result.Append('\n');
                result.Append(' ', numSpaces);
                result.Append(item);
            }

            return result.ToString();
        }

        public static bool CheckSha1(string filename, string sha1_hash)
        {
            using (var stream = File.OpenRead(filename))
            {
                var hash = SHA1.Create().ComputeHash(stream);
                var hashString = Encoding.UTF8.GetString(hash, 0, hash.Length);
                if (hashString == sha1_hash)
                    return true;
            }

            return true;
        }

        public static string Download(string url, string path = "", bool overwrite = false, string sha1_hash = "",
            bool verify_ssl = true)
        {
            if (!verify_ssl)
                Logger.Warning("Unverified HTTPS request is being made (verify_ssl=False). " +
                               "Adding certificate verification is strongly advised.");

            if (string.IsNullOrWhiteSpace(path))
                path = "./";


            if(!overwrite && File.Exists(path))
            {
                if (sha1_hash != "")
                    if (!CheckSha1(path, sha1_hash))
                        throw new Exception("File hash not matching");

                return path;
            }

            using (var client = new WebClient())
            {
                var ur = new Uri(url);
                // client.Credentials = new NetworkCredential("username", "password");
                client.DownloadProgressChanged += WebClientDownloadProgressChanged;
                client.DownloadFileCompleted += WebClientDownloadCompleted;
                Console.WriteLine(@"Downloading file:" + url);
                client.DownloadFileAsync(ur, path);
                _semaphore.Wait();
            }

            if (sha1_hash != "")
                if (!CheckSha1(path, sha1_hash))
                    throw new Exception("File hash not matching");

            return path;
        }

        private static void WebClientDownloadCompleted(object sender, AsyncCompletedEventArgs e)
        {
            var _result = !e.Cancelled;
            if (!_result) Console.Write(e.Error.ToString());

            Console.WriteLine(Environment.NewLine + "Download finished!");
            _semaphore.Release();
        }

        private static void WebClientDownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            Console.Write("\r     -->    {0}%.", e.ProgressPercentage);
        }

        public static string GetRepoUrl()
        {
            var default_repo = "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/";
            var repo_url = Environment.GetEnvironmentVariable("MXNET_GLUON_REPO");
            repo_url = !string.IsNullOrEmpty(repo_url) ? repo_url : default_repo;
            if (!repo_url.EndsWith("/"))
                repo_url = repo_url + "/";

            return repo_url;
        }

        public static string GetRepoFileUrl(string @namespace, string filename)
        {
            return string.Format("{0}{1}/{2}", GetRepoUrl(), @namespace, filename);
        }

        public static string BriefPrintList<T>(List<T> lst, int limit = 7)
        {
            var counter = 0;
            var sb = new StringBuilder();
            foreach (var item in lst)
            {
                if (counter == 7)
                {
                    sb.AppendLine(", ...,");
                    counter = 0;
                }

                sb.AppendFormat("'{0}'", item.ToString());
                counter++;
            }

            return sb.ToString();
        }

        public static bool ShapeIsKnown(Shape shape)
        {
            if (shape == null)
                return false;

            if (shape.Dimension == 0)
                return false;

            for (var i = 0; i < shape.Dimension; i++)
                if (shape[i] == 0)
                    return false;

            return true;
        }
    }
}