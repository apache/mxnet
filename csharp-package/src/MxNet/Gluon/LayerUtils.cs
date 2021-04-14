using MxNet.Gluon;
using MxNet.Gluon.NN;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Reflection;

namespace MxNet.Gluon
{
    public class LayerUtils
    {
        public static HybridBlock NormLayer(string layer_name, FuncArgs kwargs)
        {
            string typeName = $"MxNet.Gluon.NN.{layer_name}";
            var normLayer = (TypeInfo)Type.GetType(typeName, true, true);
            if (kwargs == null)
                kwargs = new FuncArgs();
            var constructor = normLayer.DeclaredConstructors.FirstOrDefault();
            var constructorParams = constructor.GetParameters();
            List<object> args = new List<object>();
            foreach (var item in constructorParams)
            {
                if (kwargs.Contains(item.Name))
                {
                    args.Add(kwargs[item.Name]);
                }
                else
                {
                    args.Add(item.DefaultValue);
                }
            }


            return (HybridBlock)Activator.CreateInstance(normLayer, args.ToArray());
        }
    }
}
