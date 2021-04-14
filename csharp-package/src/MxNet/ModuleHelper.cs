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
using System.Linq;

namespace MxNet
{
    internal class Assert
    {
        public static void IsNull(string name, object obj, string message = "")
        {
            if (obj == null)
                throw new ArgumentException(string.IsNullOrWhiteSpace(message) ? name : message);
        }

        public static void IsEqual(string name, object obj, object obj1, string message = "")
        {
            if (obj != obj1)
                throw new ArgumentException(string.IsNullOrWhiteSpace(message) ? name : message);
        }

        public static void InList(string name, string value, string[] options, string message = "")
        {
            if (!options.Contains(value))
                throw new ArgumentException(string.IsNullOrWhiteSpace(message)
                    ? $"{name} is not in {string.Join(",", options)}"
                    : message);
        }

        public static void InList(string name, int value, int[] options, string message = "")
        {
            if (!options.Contains(value))
                throw new ArgumentException(string.IsNullOrWhiteSpace(message)
                    ? $"{name} is not in {string.Join(",", options)}"
                    : message);
        }

        public static void InList(string name, uint value, uint[] options, string message = "")
        {
            if (!options.Contains(value))
                throw new ArgumentException(string.IsNullOrWhiteSpace(message)
                    ? $"{name} is not in {string.Join(",", options)}"
                    : message);
        }
    }
}