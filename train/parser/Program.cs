using System;
using System.IO;
using System.Text.RegularExpressions;

namespace ConsoleApplication
{
    public class Program
    {
        public static void Main(string[] args)
        {
            //arg1: file
            //arg2: num machines
            const int batches = 100;
            var texts = File.ReadAllLines(args[0]);
            var machines = int.Parse(args[1]);
            double totalTime = 0;
            Regex reg = new Regex("Time cost=(\\d*\\.?\\d+)");
            int cnt = 0;
            foreach(var line in texts)
            {
                Match mtch = reg.Match(line);
                if(mtch.Success)
                {
                    totalTime += double.Parse(mtch.Groups[1].Value);
                    cnt++;
                }
            }
            Console.WriteLine("{0} samples {1} average.",cnt,totalTime / (cnt * 10));
        }
    }
}
