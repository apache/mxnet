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
using System.IO.Compression;

namespace MxNet.Gluon.ModelZoo
{
    public class ModelStore
    {
        private static readonly Dictionary<string, string> model_sha1 = new Dictionary<string, string>
        {
            {"alexnet", "44335d1f0046b328243b32a26a4fbd62d9057b45"},
            {"densenet121", "f27dbf2dbd5ce9a80b102d89c7483342cd33cb31"},
            {"densenet161", "b6c8a95717e3e761bd88d145f4d0a214aaa515dc"},
            {"densenet169", "2603f878403c6aa5a71a124c4a3307143d6820e9"},
            {"densenet201", "1cdbc116bc3a1b65832b18cf53e1cb8e7da017eb"},
            {"inceptionv3", "ed47ec45a937b656fcc94dabde85495bbef5ba1f"},
            {"mobilenet0.25", "9f83e440996887baf91a6aff1cccc1c903a64274"},
            {"mobilenet0.5", "8e9d539cc66aa5efa71c4b6af983b936ab8701c3"},
            {"mobilenet0.75", "529b2c7f4934e6cb851155b22c96c9ab0a7c4dc2"},
            {"mobilenet1.0", "6b8c5106c730e8750bcd82ceb75220a3351157cd"},
            {"mobilenetv2_1.0", "36da4ff1867abccd32b29592d79fc753bca5a215"},
            {"mobilenetv2_0.75", "e2be7b72a79fe4a750d1dd415afedf01c3ea818d"},
            {"mobilenetv2_0.5", "aabd26cd335379fcb72ae6c8fac45a70eab11785"},
            {"mobilenetv2_0.25", "ae8f9392789b04822cbb1d98c27283fc5f8aa0a7"},
            {"resnet18_v1", "a0666292f0a30ff61f857b0b66efc0228eb6a54b"},
            {"resnet34_v1", "48216ba99a8b1005d75c0f3a0c422301a0473233"},
            {"resnet50_v1", "0aee57f96768c0a2d5b23a6ec91eb08dfb0a45ce"},
            {"resnet101_v1", "d988c13d6159779e907140a638c56f229634cb02"},
            {"resnet152_v1", "671c637a14387ab9e2654eafd0d493d86b1c8579"},
            {"resnet18_v2", "a81db45fd7b7a2d12ab97cd88ef0a5ac48b8f657"},
            {"resnet34_v2", "9d6b80bbc35169de6b6edecffdd6047c56fdd322"},
            {"resnet50_v2", "ecdde35339c1aadbec4f547857078e734a76fb49"},
            {"resnet101_v2", "18e93e4f48947e002547f50eabbcc9c83e516aa6"},
            {"resnet152_v2", "f2695542de38cf7e71ed58f02893d82bb409415e"},
            {"squeezenet1.0", "264ba4970a0cc87a4f15c96e25246a1307caf523"},
            {"squeezenet1.1", "33ba0f93753c83d86e1eb397f38a667eaf2e9376"},
            {"vgg11", "dd221b160977f36a53f464cb54648d227c707a05"},
            {"vgg11_bn", "ee79a8098a91fbe05b7a973fed2017a6117723a8"},
            {"vgg13", "6bc5de58a05a5e2e7f493e2d75a580d83efde38c"},
            {"vgg13_bn", "7d97a06c3c7a1aecc88b6e7385c2b373a249e95e"},
            {"vgg16", "e660d4569ccb679ec68f1fd3cce07a387252a90a"},
            {"vgg16_bn", "7f01cf050d357127a73826045c245041b0df7363"},
            {"vgg19", "ad2f660d101905472b83590b59708b71ea22b2e5"},
            {"vgg19_bn", "f360b758e856f1074a85abd5fd873ed1d98297c3"}
        };

        private static readonly string apache_repo_url = "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/";
        private static readonly string _url_format = "{0}gluon/models/{1}.zip";

        public static string ShortHash(string name)
        {
            if (model_sha1.ContainsKey(name))
                return model_sha1[name].Substring(0, 8);

            throw new Exception($"Pretrained model for {name} is not available");
        }

        public static string GetModelFile(string name, string root = "")
        {
            if (string.IsNullOrWhiteSpace(root))
                root = mx.AppPath + "\\Models";

            if (!Directory.Exists(root))
                Directory.CreateDirectory(root);

            var file_name = $"{name}-{ShortHash(name)}";
            var file_path = $"{root}\\{file_name}.params";
            var shal1_hash = model_sha1[name];
            if (File.Exists(file_path))
            {
                if (Utils.CheckSha1(file_path, shal1_hash))
                    return file_path;
                Logger.Warning("Mismatch in the content of model file detected. Downloading again.");
            }
            else
            {
                Logger.Info($"Model file not found. Downloading to {file_path}");
            }

            var zip_file_path = Path.Combine(root, file_name + ".zip");
            var repo_url = Environment.GetEnvironmentVariable("MXNET_GLUON_REPO");
            if (string.IsNullOrWhiteSpace(repo_url))
                repo_url = apache_repo_url;

            if (!repo_url.EndsWith("/"))
                repo_url += "/";

            Utils.Download(string.Format(_url_format, repo_url, file_name), zip_file_path, true);
            ZipFile.ExtractToDirectory(zip_file_path, root);
            File.Delete(zip_file_path);

            if (Utils.CheckSha1(file_path, shal1_hash))
                return file_path;
            throw new Exception("Downloaded file has different hash. Please try again.");
        }

        public static void Purge(string root = "")
        {
            if (string.IsNullOrWhiteSpace(root))
                root = mx.AppPath + "\\Models";
            var dir = new DirectoryInfo(root);
            var files = dir.GetFiles("*.params");
            foreach (var item in files) item.Delete();
        }
    }
}