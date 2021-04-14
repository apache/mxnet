using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Libs
{
    public class ContextVar<T>
    {
        private T defaultVal;
        private string name;
        private T value;

        public ContextVar(string name, T @default = default(T))
        {
            this.name = name;
            defaultVal = @default;
            value = @default;
        }

        public T Get()
        {
            return value;
        }

        public T Set(T value)
        {
            this.value = value;
            return this.value;
        }

        public void Reset(T value)
        {
            this.value = value;
        }
    }
}
