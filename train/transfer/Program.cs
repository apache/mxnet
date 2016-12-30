using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace ConsoleApplication
{
    public class Program
    {
        public static void Main(string[] args)
        {
            if(args.Length < 2)
            {
                Console.WriteLine("<hosts> <file1> [file2] ... ");
                return;
            }
            var hosts = args[0];
            var files = args.Skip(1);
            var cmd = "";
            foreach(var host in File.ReadAllLines(hosts))
            {
                var command = $"scp ";
                foreach(var file in files)
                {
                    var invoke = $"scp {file} ubuntu@{host}:{file}\n";
                    cmd += invoke;
                    // var procInfo = new ProcessStartInfo(invoke);
                    // procInfo.RedirectStandardError = true;
                    // procInfo.RedirectStandardOutput = true;
                    // Console.Write("Executing {0}...", invoke);
                    // var process = Process.Start(procInfo);
                    // process.WaitForExit();
                    // var stdErr = process.StandardError.ReadToEnd();
                    // var stdOutput = process.StandardOutput.ReadToEnd();
                    // if(stdErr!=String.Empty)
                    // {
                    //     Console.ForegroundColor = ConsoleColor.Red;
                    //     Console.WriteLine("\n{0}", stdErr);
                    //     Console.ForegroundColor = ConsoleColor.Gray;
                    // }
                    // if(stdOutput!=String.Empty)
                    // {
                    //     Console.WriteLine("\n{0}", stdOutput);
                    // }
                    // Console.WriteLine("Done...");
                    File.WriteAllText("run.sh", cmd);
                }
            }
        }
    }
}
