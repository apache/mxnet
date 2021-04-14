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
using MxNet.Interop;
using MxNet.Optimizers;
using Newtonsoft.Json;

namespace MxNet.KVstore
{
    public class KVStoreServer
    {
        public delegate void ServerController(int cmd_id, string cmd_body);

        private IntPtr handle;
        private bool init_logging;
        private readonly KVStore kvstore;

        public KVStoreServer(KVStore kvstore)
        {
            this.kvstore = kvstore;
            handle = kvstore.handle;
            init_logging = false;
        }

        public ServerController Controller()
        {
            ServerController ctl = (cmd_id, cmd_body) =>
            {
                var head = "";
                if (!init_logging)
                {
                    head = string.Format("{0} Server[{1}]", DateTime.Now.ToString(), kvstore.Rank);
                    Logger.Log(head);
                    init_logging = true;
                }

                if (cmd_id == 0)
                {
                    var optimizer = JsonConvert.DeserializeObject(cmd_body);
                    kvstore.SetOptimizer((Optimizer) optimizer);
                }
                else
                {
                    Console.WriteLine("Server {0}, unknown command ({1},{2})", kvstore.Rank, cmd_id, cmd_body);
                }
            };

            return ctl;
        }

        public void Run()
        {
            var serverController = Controller();
            var controller = new MXKVStoreServerController
            {
                body = "",
                head = 0,
                controller_handle = serverController.Method.MethodHandle.GetFunctionPointer()
            };

            NativeMethods.MXKVStoreRunServer(kvstore.handle, controller, controller.controller_handle);
        }

        public static void InitServerModule()
        {
            NativeMethods.MXKVStoreIsWorkerNode(out var is_worker);
            if (is_worker == 0)
            {
                var kvstore = KVStoreBase.Create("dist");
                var server = new KVStoreServer(kvstore);
                server.Run();
            }
        }
    }
}