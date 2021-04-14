using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace MxNet
{
    public class Visualization
    {
        private static int[] _str2tuple(string input)
        {
            var matchCollection = Regex.Matches(input, @"\d+");
            List<int> result = new List<int>();
            foreach (Match item in matchCollection)
            {
                result.Add(Convert.ToInt32(item.Value));
            }

            return result.ToArray();
        }

        public static void PrintSummary(Symbol symbol, Dictionary<string, Shape> shape = null, int line_length = 120, List<double> positions = null)
        {
            if (positions == null) {
                positions = new List<double> {
                                            0.44,
                                            0.64,
                                            0.74,
                                            1.0
                                        };
            }

            var show_shape = false;
            Dictionary<string, Shape> shape_dict = new Dictionary<string, Shape>();
            if (shape != null)
            {
                show_shape = true;
                var interals = symbol.GetInternals();
                var _tup_1 = interals.InferShape(shape);
                var out_shapes = _tup_1.Item2;
                if (out_shapes == null)
                {
                    throw new Exception("Input shape is incomplete");
                }

                
                var outputNames = interals.ListOutputs();
                for(int i = 0; i<outputNames.Count; i++)
                {
                    shape_dict.Add(outputNames[i], out_shapes[i]);
                }
            }

            dynamic conf = JsonConvert.DeserializeObject<ExpandoObject>(symbol.ToJSON());
            dynamic nodes = conf.nodes;
            dynamic heads = new HashSet<object>(conf.heads[0]);
            if (positions.Last() <= 1)
            {
                positions = (from p in positions
                             select Convert.ToDouble(Convert.ToInt32(line_length * p))).ToList();
            }

            // header names for the different log elements
            var to_display = new List<string> {
                                                "Layer (type)",
                                                "Output Shape",
                                                "Param #",
                                                "Previous Layer"
                                            };

            Action<string[], double[]> print_row = (fields, _positions) => {
                var line = "";
                foreach (var i in Enumerable.Range(0, fields.Length))
                {
                    var field = fields[i];
                    line += field;
                    line = line.Substring(0, Convert.ToInt32(_positions[i]));
                    var lineCharCount = Convert.ToInt32(_positions[i] - line.Length);
                    foreach (var item in Enumerable.Range(0, lineCharCount))
                    {
                        line += " ";
                    }
                }

                Console.WriteLine(line);
            };

            string msg = "";
            foreach (var item in Enumerable.Range(0, line_length))
            {
                msg += "_";
            }

            Console.WriteLine(msg);
            print_row(to_display.ToArray(), positions.ToArray());

            msg = "";
            foreach (var item in Enumerable.Range(0, line_length))
            {
                msg += "=";
            }
            
            Console.WriteLine(msg);

            Func<dynamic, Shape, int> print_layer_summary = (node, out_shape) => {
                string first_connection;
                int num_group;
                string key;
                dynamic op = node.op;
                var pre_node = new List<string>();
                var pre_filter = 0;
                if (op != "null")
                {
                    var inputs = node["inputs"];
                    foreach (var item in inputs)
                    {
                        var input_node = nodes[item[0]];
                        var input_name = input_node["name"];
                        if (input_node["op"] != "null" || heads.Contains(item[0]))
                        {
                            // add precede
                            pre_node.Add(input_name);
                            if (show_shape)
                            {
                                if (input_node["op"] != "null")
                                {
                                    key = input_name + "_output";
                                }
                                else
                                {
                                    key = input_name;
                                }
                                if (shape_dict.ContainsKey(key))
                                {
                                    var shape1 = shape_dict[key];
                                    pre_filter = pre_filter + Convert.ToInt32(shape1[0]);
                                }
                            }
                        }
                    }
                }

                var cur_param = 0;
                if (op == "Convolution")
                {
                    if (node["attrs"].Contains("no_bias") && node["attrs"]["no_bias"] == "True")
                    {
                        num_group = Convert.ToInt32(node["attrs"].get("num_group", "1"));
                        cur_param = pre_filter * Convert.ToInt32(node["attrs"]["num_filter"]) / num_group;
                        foreach (var k in _str2tuple(node["attrs"]["kernel"]))
                        {
                            cur_param *= Convert.ToInt32(k);
                        }
                    }
                    else
                    {
                        num_group = Convert.ToInt32(node["attrs"].get("num_group", "1"));
                        cur_param = pre_filter * Convert.ToInt32(node["attrs"]["num_filter"]) / num_group;
                        foreach (var k in _str2tuple(node["attrs"]["kernel"]))
                        {
                            cur_param *= Convert.ToInt32(k);
                        }
                        cur_param += Convert.ToInt32(node["attrs"]["num_filter"]);
                    }
                }
                else if (op == "FullyConnected")
                {
                    if (node["attrs"].Contains("no_bias") && node["attrs"]["no_bias"] == "True")
                    {
                        cur_param = pre_filter * Convert.ToInt32(node["attrs"]["num_hidden"]);
                    }
                    else
                    {
                        cur_param = (pre_filter + 1) * Convert.ToInt32(node["attrs"]["num_hidden"]);
                    }
                }
                else if (op == "BatchNorm")
                {
                    key = node["name"] + "_output";
                    if (show_shape)
                    {
                        var num_filter = shape_dict[key][1];
                        cur_param = Convert.ToInt32(num_filter) * 2;
                    }
                }
                else if (op == "Embedding")
                {
                    cur_param = Convert.ToInt32(node["attrs"]["input_dim"]) * Convert.ToInt32(node["attrs"]["output_dim"]);
                }

                if (pre_node == null)
                {
                    first_connection = "";
                }
                else
                {
                    first_connection = pre_node[0];
                }

                var fields = new List<string> {
                                                node["name"] + "(" + op + ")",
                                                string.Join("x", out_shape),
                                                cur_param.ToString(),
                                                first_connection
                                            };
                print_row(fields.ToArray(), positions.ToArray());
                if (pre_node.Count > 1)
                {
                    foreach (var i in Enumerable.Range(1, pre_node.Count - 1))
                    {
                        fields = new List<string> {
                        "",
                        "",
                        "",
                        pre_node[i]
                    };
                        print_row(fields.ToArray(), positions.ToArray());
                    }
                }

                return cur_param;
            };

            var total_params = 0;
            for(int i = 0; i< nodes.Length; i++)
            {
                var node = nodes[i];
                var out_shape = new Shape();
                var op = node["op"];
                string key = "";

                if (op == "null" && i > 0)
                {
                    continue;
                }
                if (op != "null" || heads.Contains(i))
                {
                    if (show_shape)
                    {
                        if (op != "null")
                        {
                            key = node["name"] + "_output";
                        }
                        else
                        {
                            key = node["name"];
                        }
                        if (shape_dict.ContainsKey(key))
                        {
                            out_shape = shape_dict[key];
                        }
                    }
                }

                total_params += print_layer_summary(nodes[i], out_shape);
                if (i == nodes.Count - 1)
                {
                    msg = "";
                    foreach (var item in Enumerable.Range(0, line_length))
                    {
                        msg += "=";
                    }

                    Console.WriteLine(msg);
                }
                else
                {
                    msg = "";
                    foreach (var item in Enumerable.Range(0, line_length))
                    {
                        msg += "_";
                    }

                    Console.WriteLine(msg);
                }
            }

            Console.WriteLine($"Total params: {total_params}");
            msg = "";
            foreach (var item in Enumerable.Range(0, line_length))
            {
                msg += "_";
            }

            Console.WriteLine(msg);
        }
    }
}
