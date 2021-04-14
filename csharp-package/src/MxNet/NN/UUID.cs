using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN
{
    public class UUID
    {
        private static Dictionary<string, int> CurrentIndexes = new Dictionary<string, int>();

        public static void Reset()
        {
            CurrentIndexes = new Dictionary<string, int>();
        }

        private static int Next(string name)
        {
            if(!CurrentIndexes.ContainsKey(name))
            {
                CurrentIndexes.Add(name, 0);
            }

            CurrentIndexes[name] += 1;
            return CurrentIndexes[name];
        }

        public static string GetID(string name)
        {
            return string.Format("{0}_{1}", name.ToLower(), Next(name));
        }
    }
}
