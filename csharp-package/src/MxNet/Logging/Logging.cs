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
using System.Runtime.InteropServices;
using MxNet.Interop;

// ReSharper disable once CheckNamespace
namespace MxNet
{
    public static class Logging
    {
        #region Constructors

        static Logging()
        {
            var symbols = new[]
            {
                new {Operator = Operator.Lesser, Symbol = "<"},
                new {Operator = Operator.Greater, Symbol = ">"},
                new {Operator = Operator.LesserEqual, Symbol = "<="},
                new {Operator = Operator.GreaterEqual, Symbol = ">="},
                new {Operator = Operator.Equal, Symbol = "=="},
                new {Operator = Operator.NotEqual, Symbol = "!="}
            };

            OperatorSymbols = new string[symbols.Length];
            for (var index = 0; index < symbols.Length; index++)
                OperatorSymbols[index] = symbols[index].Symbol;
        }

        #endregion

        private enum Operator
        {
            Lesser = 0,

            Greater,

            LesserEqual,

            GreaterEqual,

            Equal,

            NotEqual
        }

        #region Fields

        private static readonly string[] OperatorSymbols;

        public static bool ThrowException { get; set; } = true;

        #endregion

        #region Methods

        public static void CHECK<T>(T x, string msg = "")
            where T : class
        {
            if (x != null)
                return;

            var message = string.IsNullOrEmpty(msg) ? $"Check failed: {x} " : $"Check failed: {x} {msg}";
            LOG_FATAL(message);
        }

        public static void CHECK(bool x, string msg = "")
        {
            if (x)
                return;

            var message = string.IsNullOrEmpty(msg) ? $"Check failed: {x} " : $"Check failed: {x} {msg}";
            LOG_FATAL(message);
        }

        private static void CHECK_EQ(string x, string y, string msg = "")
        {
            var error = NativeMethods.MXGetLastError();
            string message;
            if (string.IsNullOrEmpty(msg))
                message =
                    $"Check failed: {x} {OperatorSymbols[(int) Operator.Equal]} {y} {Marshal.PtrToStringAnsi(error) ?? ""}";
            else
                message = $"Check failed: {x} {OperatorSymbols[(int) Operator.Equal]} {y} {msg}";

            LOG_FATAL(message);
        }

        public static void CHECK_EQ(int x, int y, string msg = "")
        {
            // dmlc-core/include/dmlc/logging.h
            if (x == y)
                return;

            CHECK_EQ(x.ToString(), y.ToString(), msg);
        }

        public static void CHECK_EQ(uint x, uint y, string msg = "")
        {
            // dmlc-core/include/dmlc/logging.h
            if (x == y)
                return;

            CHECK_EQ(x.ToString(), y.ToString(), msg);
        }

        public static void CHECK_EQ(bool x, bool y, string msg = "")
        {
            // dmlc-core/include/dmlc/logging.h
            if (x == y)
                return;

            CHECK_EQ(x.ToString(), y.ToString(), msg);
        }

        public static void CHECK_EQ(Shape x, Shape y, string msg = "")
        {
            // dmlc-core/include/dmlc/logging.h
            if (x == y)
                return;

            CHECK_EQ(x.ToString(), y.ToString(), msg);
        }

        public static void CHECK_NE(int x, int y)
        {
            // dmlc-core/include/dmlc/logging.h
            if (x != y)
                return;

            var error = NativeMethods.MXGetLastError();
            var message =
                $"Check failed: {x} {OperatorSymbols[(int) Operator.NotEqual]} {y} {Marshal.PtrToStringAnsi(error) ?? ""}";
            LOG_FATAL(message);
        }

        public static void LG(string message)
        {
            Console.WriteLine(message);
        }

        #region Helpers

        private static void LOG_FATAL(string message)
        {
            if (ThrowException)
                throw new MXNetException(message);

            Console.WriteLine(message);
        }

        #endregion

        #endregion
    }
}