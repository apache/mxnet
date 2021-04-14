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
using System.Diagnostics;

namespace MxNet
{
    public class Logger : IDisposable
    {
        private static TextWriterTraceListener trace;
        private string filename = "";
        private string name = "";

        public void Dispose()
        {
            trace.Close();
            trace.Dispose();
        }

        public static void Log(string message, TraceLevel level = TraceLevel.Verbose)
        {
            if (trace != null)
                trace.Write(Formatter.FormatMessage(message, level));

            Console.ForegroundColor = Formatter.GetColor(level);
            Console.WriteLine(message);
            Console.ResetColor();
        }

        public static void Warning(string message)
        {
            Log(message, TraceLevel.Warning);
        }

        public static void Info(string message)
        {
            Log(message, TraceLevel.Info);
        }

        public static void Error(string message)
        {
            Log(message, TraceLevel.Error);
        }

        public static void Configure(string filename, string name = "")
        {
            if (!string.IsNullOrWhiteSpace(name))
                trace = new TextWriterTraceListener(filename, name);
            else
                trace = new TextWriterTraceListener(filename);
        }

        public class Formatter
        {
            public static ConsoleColor GetColor(TraceLevel level)
            {
                var color = ConsoleColor.White;

                switch (level)
                {
                    case TraceLevel.Error:
                        color = ConsoleColor.Red;
                        break;
                    case TraceLevel.Warning:
                        color = ConsoleColor.DarkYellow;
                        break;
                    case TraceLevel.Info:
                    case TraceLevel.Verbose:
                        color = ConsoleColor.White;
                        break;
                }

                return color;
            }

            public static string GetLabel(TraceLevel level)
            {
                switch (level)
                {
                    case TraceLevel.Error:
                        return "ERROR";
                    case TraceLevel.Warning:
                        return "WARNING";
                    case TraceLevel.Info:
                        return "INFO";
                    case TraceLevel.Verbose:
                        return "VERBOSE";
                }

                return "I";
            }

            public static string FormatMessage(string message, TraceLevel level)
            {
                return string.Format("{0}: {1}", GetLabel(level), message);
            }
        }
    }
}