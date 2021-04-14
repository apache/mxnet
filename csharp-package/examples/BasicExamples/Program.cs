using MxNet;
using MxNet.Numpy;
using System;
using System.Linq;
using System.Text;


namespace BasicExamples
{
    class Program
    {
        static void Main(string[] args)
        {
            //Console.WriteLine("Runnin XOR Example......");
            //XORGate.Run();
            //CrashCourse_NN.Run();
            //LogisticRegressionExplained.Run();
            var methods = mx.GetAllRegisteredOperators();
            var y = np.full(new Shape(3, 3), 0.6);
            var x = np.random.power(y, new Shape(3, 3));
            
            var z = np.linalg.cholesky(x);
        }

        private static void GenerateFOps()
        {
            var methods = typeof(F).GetMethods();
            StringBuilder fclass = new StringBuilder();
            foreach (var method in methods)
            {
                var parameters = method.GetParameters();
                string paramstr = "";
                string ndcall = $"nd_np_ops.{method.Name}(";
                string symcall = $"sym_np_ops.{method.Name}(";
                bool is_symbol = false;
                if (parameters.Length > 0)
                {
                    foreach (var item in parameters)
                    { 
                        if (item.ParameterType.Name == "is_symbol")
                        {
                            is_symbol = true;
                            continue;
                        }

                        if (item.ParameterType.Name == "NDArray")
                        {
                            paramstr += $"NDArrayOrSymbol {item.Name},";
                            ndcall += $"{item.Name},";
                            symcall += $"{item.Name},";
                        }
                        else if (item.ParameterType.Name == "NDArrayList")
                        {
                            paramstr += $"NDArrayOrSymbolList {item.Name},";
                            ndcall += $"{item.Name}.NDArrays,";
                            symcall += $"{item.Name}.Symbols,";
                        }
                        else if (item.ParameterType.Name == "Nullable`1")
                        {
                            paramstr += $"{item.ParameterType.GenericTypeArguments[0].Name}? {item.Name},";
                            ndcall += $"{item.Name},";
                            symcall += $"{item.Name},";
                        }
                        else if (item.ParameterType.Name == "Tuple`1")
                        {
                            paramstr += $"Tuple<{item.ParameterType.GenericTypeArguments[0].Name}> {item.Name},";
                            ndcall += $"{item.Name},";
                            symcall += $"{item.Name},";
                        }
                        else
                        {
                            paramstr += $"{item.ParameterType.Name} {item.Name},";
                            ndcall += $"{item.Name},";
                            symcall += $"{item.Name},";
                        }
                    }

                    paramstr = paramstr.Remove(paramstr.LastIndexOf(','));
                    ndcall = ndcall.Remove(ndcall.LastIndexOf(',')) + ");";
                    symcall = symcall.Remove(symcall.LastIndexOf(',')) + ");";
                }

                string returnType = method.ReturnType.Name;
                if (returnType == "NDArray")
                    returnType = "NDArrayOrSymbol";
                else if (returnType == "NDArrayList")
                    returnType = "NDArrayOrSymbolList";

                string methodBody = $"public static {returnType} {method.Name}({paramstr})";
                methodBody += "\n{";

                string firstNdParam = "";
                var ndparam = parameters.FirstOrDefault(x => x.ParameterType.Name == "NDArrayOrSymbol");
                if (ndparam == null)
                {
                    ndparam = parameters.FirstOrDefault(x => x.ParameterType.Name == "NDArrayOrSymbolList");
                }

                if (is_symbol)
                {
                    methodBody += $"if (!is_symbol)";
                    methodBody += "\n{\n";
                    methodBody += "return " + ndcall + "\n}\n";
                    methodBody += "return " + symcall;
                    methodBody += "\n}";

                    fclass.AppendLine(methodBody);
                }
                else if (ndparam != null)
                {
                    firstNdParam = ndparam.ParameterType.Name == "NDArrayList" ? $"{ndparam.Name}[0]" : ndparam.Name;

                    methodBody += $"if ({firstNdParam}.IsNDArray)";
                    methodBody += "\n{\n";
                    methodBody += "return " + ndcall + "\n}\n";
                    methodBody += "return " + symcall;
                    methodBody += "\n}";

                    fclass.AppendLine(methodBody);
                }
            }

            string all = fclass.ToString();
        }
    }
}
