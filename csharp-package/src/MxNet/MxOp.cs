using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet
{
    public class MxOp
    {
        public string Name { get; set; }

        public List<MxOpArg> Args { get; set; }

        public MxOp(string name, List<MxOpArg> args)
        {
            Name = name;
            Args = args;
        }

        public override string ToString()
        {
            return Name;
        }
    }

    public class MxOpArg
    {
        public string Name { get; set; }

        public string DataType { get; set; }

        public MxOpArg(string name, string dataType)
        {
            Name = name;
            DataType = dataType;
        }

        public override string ToString()
        {
            return $"{DataType} {Name}";
        }
    }
}
