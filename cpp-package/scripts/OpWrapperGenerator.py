# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# -*- coding: utf-8 -*-
# This is a python script that generates operator wrappers such as FullyConnected,
# based on current libmxnet.dll. This script is written so that we don't need to
# write new operator wrappers when new ones are added to the library.

import codecs
import filecmp
import logging
import os
import platform
import re
import shutil
import sys
import tempfile
import io
from ctypes import *
from ctypes.util import find_library


def gen_enum_value(value):
    return 'k' + value[0].upper() + value[1:]


class EnumType:
    name = ''
    enumValues = []

    def __init__(self, typeName='ElementWiseOpType',
                 typeString="{'avg', 'max', 'sum'}"):
        self.name = typeName

        if (typeString[0] == '{'):  # is a enum type
            isEnum = True
            # parse enum
            self.enumValues = typeString[typeString.find(
                '{') + 1:typeString.find('}')].split(',')

            for i in range(0, len(self.enumValues)):
                self.enumValues[i] = self.enumValues[i].strip().strip("'")
        else:
            logging.warn(
                "trying to parse none-enum type as enum: %s" % typeString)

    def GetDefinitionString(self, indent=0):
        indentStr = ' ' * indent
        ret = indentStr + 'enum class %s {\n' % self.name

        for i in range(0, len(self.enumValues)):
            ret = ret + indentStr + \
                '  %s = %d' % (gen_enum_value(self.enumValues[i]), i)

            if (i != len(self.enumValues) - 1):
                ret = ret + ","
            ret = ret + "\n"
        ret = ret + "};\n"

        return ret

    def GetDefaultValueString(self, value=''):
        return self.name + "::" + gen_enum_value(value)

    def GetEnumStringArray(self, indent=0):
        indentStr = ' ' * indent
        ret = indentStr + 'static const char *%sValues[] = {\n' % self.name

        for i in range(0, len(self.enumValues)):
            ret = ret + indentStr + '  "%s"' % self.enumValues[i]

            if (i != len(self.enumValues) - 1):
                ret = ret + ","
            ret = ret + "\n"
        ret = ret + indentStr + "};\n"

        return ret

    def GetConvertEnumVariableToString(self, variable=''):
        return "%sValues[int(%s)]" % (self.name, variable)


class Arg:
    typeDict = {'boolean': 'bool',
                'boolean or None': 'dmlc::optional<bool>',
                'Shape(tuple)': 'Shape',
                'Symbol': 'Symbol',
                'NDArray': 'Symbol',
                'NDArray-or-Symbol': 'Symbol',
                'Symbol[]': 'const std::vector<Symbol>&',
                'Symbol or Symbol[]': 'const std::vector<Symbol>&',
                'NDArray[]': 'const std::vector<Symbol>&',
                'caffe-layer-parameter': '::caffe::LayerParameter',
                'NDArray-or-Symbol[]': 'const std::vector<Symbol>&',
                'float': 'mx_float',
                'real_t': 'mx_float',
                'int': 'int',
                'int (non-negative)': 'uint32_t',
                'long (non-negative)': 'uint64_t',
                'int or None': 'dmlc::optional<int>',
                'int64 or None': 'dmlc::optional<int64_t>',
                'float or None': 'dmlc::optional<float>',
                'long': 'int64_t',
                'double': 'double',
                'double or None': 'dmlc::optional<double>',
                'Shape or None': 'dmlc::optional<Shape>',
                'string': 'const std::string&',
                'tuple of <float>': 'nnvm::Tuple<mx_float>'}
    name = ''
    type = ''
    description = ''
    isEnum = False
    enum = None
    hasDefault = False
    defaultString = ''

    def __init__(self, opName='', argName='', typeString='', descString=''):
        self.name = argName
        self.description = descString

        if (typeString[0] == '{'):  # is enum type
            self.isEnum = True
            self.enum = EnumType(self.ConstructEnumTypeName(
                opName, argName), typeString)
            self.type = self.enum.name
        else:
            try:
                self.type = self.typeDict[typeString.split(',')[0]]
            except:
                print('argument "%s" of operator "%s" has unknown type "%s"' %
                      (argName, opName, typeString))
                pass

        if typeString.find('default=') != -1:
            self.hasDefault = True
            self.defaultString = typeString.split(
                'default=')[1].strip().strip("'")

            if typeString.startswith('string'):
                self.defaultString = self.MakeCString(self.defaultString)
            elif self.isEnum:
                self.defaultString = self.enum.GetDefaultValueString(
                    self.defaultString)
            elif self.defaultString == 'None':
                self.defaultString = self.type + '()'
            elif self.type == "bool":
                if self.defaultString == "1" or self.defaultString == "True":
                    self.defaultString = "true"
                else:
                    self.defaultString = "false"
            elif self.defaultString[0] == '(':
                self.defaultString = 'Shape' + self.defaultString
            elif self.defaultString[0] == '[':
                self.defaultString = 'Shape(' + self.defaultString[1:-1] + ")"
            elif self.type == 'dmlc::optional<int>':
                self.defaultString = self.type + '(' + self.defaultString + ')'
            elif self.type == 'dmlc::optional<bool>':
                self.defaultString = self.type + '(' + self.defaultString + ')'
            elif typeString.startswith('caffe-layer-parameter'):
                self.defaultString = 'textToCaffeLayerParameter(' + self.MakeCString(
                    self.defaultString) + ')'
                hasCaffe = True

    def MakeCString(self, str):
        str = str.replace('\n', "\\n")
        str = str.replace('\t', "\\t")

        return '\"' + str + '\"'

    def ConstructEnumTypeName(self, opName='', argName=''):
        a = opName[0].upper()
        # format ArgName so instead of act_type it returns ActType
        argNameWords = argName.split('_')
        argName = ''

        for an in argNameWords:
            argName = argName + an[0].upper() + an[1:]
        typeName = a + opName[1:] + argName

        return typeName


class Op:
    name = ''
    description = ''
    args = []

    def __init__(self, name='', description='', args=[]):
        self.name = name
        self.description = description
        # add a 'name' argument
        nameArg = Arg(self.name,
                      'symbol_name',
                      'string',
                      'name of the resulting symbol')
        args.insert(0, nameArg)
        # reorder arguments, put those with default value to the end
        orderedArgs = []

        for arg in args:
            if not arg.hasDefault:
                orderedArgs.append(arg)

        for arg in args:
            if arg.hasDefault:
                orderedArgs.append(arg)
        self.args = orderedArgs

    def WrapDescription(self, desc=''):
        ret = []
        sentences = desc.split('.')
        lines = desc.split('\n')

        for line in lines:
            line = line.strip()

            if len(line) <= 80:
                ret.append(line.strip())
            else:
                while len(line) > 80:
                    pos = line.rfind(' ', 0, 80)+1

                    if pos <= 0:
                        pos = line.find(' ')

                    if pos < 0:
                        pos = len(line)
                    ret.append(line[:pos].strip())
                    line = line[pos:]

        return ret

    def GenDescription(self, desc='',
                       firstLineHead=' * \\brief ',
                       otherLineHead=' *        '):
        ret = ''
        descs = self.WrapDescription(desc)
        ret = ret + firstLineHead

        if len(descs) == 0:
            return ret.rstrip()
        ret = (ret + descs[0]).rstrip() + '\n'

        for i in range(1, len(descs)):
            ret = ret + (otherLineHead + descs[i]).rstrip() + '\n'

        return ret

    def GetOpDefinitionString(self, use_name, indent=0):
        ret = ''
        indentStr = ' ' * indent
        # define enums if any

        for arg in self.args:
            if arg.isEnum and use_name:
                        # comments
                ret = ret + self.GenDescription(arg.description,
                                                '/*! \\brief ',
                                                ' *        ')
                ret = ret + " */\n"
                # definition
                ret = ret + arg.enum.GetDefinitionString(indent) + '\n'
        # create function comments
        ret = ret + self.GenDescription(self.description,
                                        '/*!\n * \\brief ',
                                        ' *        ')

        for arg in self.args:
            if arg.name != 'symbol_name' or use_name:
                ret = ret + self.GenDescription(arg.name + ' ' + arg.description,
                                                ' * \\param ',
                                                ' *        ')
        ret = ret + " * \\return new symbol\n"
        ret = ret + " */\n"
        # create function header
        declFirstLine = indentStr + 'inline Symbol %s(' % self.name
        ret = ret + declFirstLine
        argIndentStr = ' ' * len(declFirstLine)
        arg_start = 0 if use_name else 1

        if len(self.args) > arg_start:
            ret = ret + self.GetArgString(self.args[arg_start])

        for i in range(arg_start+1, len(self.args)):
            ret = ret + ',\n'
            ret = ret + argIndentStr + self.GetArgString(self.args[i])
        ret = ret + ') {\n'
        # create function body
        # if there is enum, generate static enum<->string mapping

        for arg in self.args:
            if arg.isEnum:
                ret = ret + arg.enum.GetEnumStringArray(indent + 2)
        # now generate code
        ret = ret + indentStr + '  return Operator(\"%s\")\n' % self.name

        for arg in self.args:   # set params
            if arg.type == 'Symbol' or \
                    arg.type == 'const std::string&' or \
                    arg.type == 'const std::vector<Symbol>&':

                continue
            v = arg.name

            if arg.isEnum:
                v = arg.enum.GetConvertEnumVariableToString(v)
            ret = ret + indentStr + ' ' * 11 + \
                '.SetParam(\"%s\", %s)\n' % (arg.name, v)
        # ret = ret[:-1]  # get rid of the last \n
        symbols = ''
        inputAlreadySet = False

        for arg in self.args:   # set inputs
            if arg.type != 'Symbol':
                continue
            inputAlreadySet = True
            # if symbols != '':
            #    symbols = symbols + ', '
            #symbols = symbols + arg.name
            ret = ret + indentStr + ' ' * 11 + \
                '.SetInput(\"%s\", %s)\n' % (arg.name, arg.name)

        for arg in self.args:   # set input arrays vector<Symbol>
            if arg.type != 'const std::vector<Symbol>&':
                continue

            if (inputAlreadySet):
                logging.error(
                    "op %s has both Symbol[] and Symbol inputs!" % self.name)
            inputAlreadySet = True
            symbols = arg.name
            ret = ret + '(%s)\n' % symbols
        ret = ret + indentStr + ' ' * 11

        if use_name:
            ret = ret + '.CreateSymbol(symbol_name);\n'
        else:
            ret = ret + '.CreateSymbol();\n'
        ret = ret + indentStr + '}\n'

        return ret

    def GetArgString(self, arg):
        ret = '%s %s' % (arg.type, arg.name)

        if arg.hasDefault:
            ret = ret + ' = ' + arg.defaultString

        return ret


blacklist = [
    '_CustomFunction',
    '_CachedOp',
    '_cvimdecode',
    '_cvimread',
    '_cvimresize',
    '_cvcopyMakeBorder',
    '_NoGradient',
    '_foreach',
    '_while_loop',
    '_cond',
    '_mp_adamw_update',
    '_adamw_update',
    'multi_lars',
    'multi_sum_sq',
    'Custom',
    '_rnn_param_concat',
    '_npi_choice',
    '_npi_multinomial',
    '_histogram',
    'BlockGrad',
    '_CrossDeviceCopy',
    '_Native',
    '_NDArray',
    'MakeLoss',
    'SVMOutput',
    '_imdecode',
    'BatchNorm_v1',
    'Convolution_v1',
    'Pooling_v1',
    'multi_all_finite',
    'khatri_rao',
    'Concat',
    'UpSampling',
    '_npi_concatenate',
    '_npi_stack',
    '_npi_vstack',
    'amp_cast',
    'amp_multicast',
    'cast_storage',
    'add_n',
    'stack',
    'topk',
    'Crop',
    'GridGenerator',
    'fill_element_0index',
]

dtype_type = "std::string"

arg_type_dict = {
    "mx_float": "double",
    "double": "double",
    "int": "int",
    "uint32_t": "int",
    "int64_t": "int64_t",
    "uint64_t": "int64_t",
    "bool": "bool",
    "Shape": "ir::Array<ir::Integer>",
    "const std::string&": "std::string",

    "dmlc::optional<int>": "int",
    "dmlc::optional<float>": "double",
    "dmlc::optional<bool>": "bool",
    "dmlc::optional<double>": "double",
    "dmlc::optional<Shape>": "ir::Array<ir::Integer>",

    "LeakyReLUActType": "std::string",
    "ActivationActType": "std::string",
    "ConvolutionCudnnTune": "std::string",
    "ConvolutionLayout": "std::string",
    "CTCLossBlankLabel": "std::string",
    "DeconvolutionCudnnTune": "std::string",
    "DeconvolutionLayout": "std::string",
    "DropoutMode": "std::string",
    "PoolingPoolType": "std::string",
    "PoolingPoolingConvention": "std::string",
    "PoolingLayout": "std::string",
    "SoftmaxActivationMode": "std::string",
    "PadMode": "std::string",
    "RNNMode": "std::string",
    "SoftmaxOutputNormalization": "std::string",
    "PickMode": "std::string",
    "DotForwardStype": "std::string",
    "Batch_dotForwardStype": "std::string",
    "TakeMode": "std::string",
    "L2NormalizationMode": "std::string",
    "SpatialTransformerTransformType": "std::string",
    "SpatialTransformerSamplerType": "std::string",

    "Log_softmaxDtype": dtype_type,
    "SoftminDtype": dtype_type,
    "SoftmaxDtype": dtype_type,
    "_np_sumDtype": dtype_type,
    "_npi_hanningDtype": dtype_type,
    "_npi_hammingDtype": dtype_type,
    "_npi_blackmanDtype": dtype_type,
    "_npi_zerosDtype": dtype_type,
    "_npi_onesDtype": dtype_type,
    "_npi_meanDtype": dtype_type,
    "_npi_stdDtype": dtype_type,
    "_npi_varDtype": dtype_type,
    "_npi_identityDtype": dtype_type,
    "_npi_arangeDtype": dtype_type,
    "_npi_indicesDtype": dtype_type,
    "_np_cumsumDtype": dtype_type,
    "_np_prodDtype": dtype_type,
    "_npi_normalDtype": dtype_type,
    "_npi_uniformDtype": dtype_type,
    "_sample_uniformDtype": dtype_type,
    "_sample_normalDtype": dtype_type,
    "_sample_gammaDtype": dtype_type,
    "_sample_exponentialDtype": dtype_type,
    "_sample_poissonDtype": dtype_type,
    "_sample_negative_binomialDtype": dtype_type,
    "_sample_generalized_negative_binomialDtype": dtype_type,
    "_sample_multinomialDtype": dtype_type,
    "_random_uniformDtype": dtype_type,
    "_random_normalDtype": dtype_type,
    "_random_gammaDtype": dtype_type,
    "_random_exponentialDtype": dtype_type,
    "_random_poissonDtype": dtype_type,
    "_random_negative_binomialDtype": dtype_type,
    "_random_generalized_negative_binomialDtype": dtype_type,
    "_random_randintDtype": dtype_type,
    "NormOutDtype": dtype_type,
    "CastDtype": dtype_type,
    "EmbeddingDtype": dtype_type,
    "One_hotDtype": dtype_type,
    "_zerosDtype": dtype_type,
    "_eyeDtype": dtype_type,
    "_onesDtype": dtype_type,
    "_fullDtype": dtype_type,
    "_arangeDtype": dtype_type,
    "_linspaceDtype": dtype_type,
    "ArgsortDtype": dtype_type,
}

op_attrs = {
}

attrs_header = open("/tmp/legacy_nnvm_attrs.h", "w")
attrs_header.write("""/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * Copyright (c) 2019 by Contributors
 * \\file legacy_nnvm_attrs.h
 * \\author Junru Shao
 */
#pragma once
#if MXNET_USE_TVM_OP && !defined MXNET_AMALGAMATION
#include <string>

#include "../../ir.h"

namespace mxnet {
namespace v3 {
namespace op {
namespace attrs {
""")

attrs_cc = open("/tmp/legacy_nnvm_attrs.cc", "w")
attrs_cc.write("""/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * Copyright (c) 2019 by Contributors
 * \\file legacy_nnvm_attrs.cc
 * \\author Junru Shao
 */
#if MXNET_USE_TVM_OP && !defined MXNET_AMALGAMATION
#include "../../include/ir.h"
#include "../../include/op/attrs/legacy_nnvm_attrs.h"

namespace mxnet {
namespace v3 {
namespace op {
namespace attrs {
namespace {
""")

def PrintAttrs(name, args):
    transformed_name = f"Legacy{name[0].upper() + name[1:]}"
    for alpha in list(range(ord('a'), ord('z') + 1)) + list(range(ord('A'), ord('Z') + 1)):
        alpha = chr(alpha)
        transformed_name = transformed_name.replace("_" + alpha, alpha.upper())
    assert "_" not in transformed_name
    assert transformed_name not in op_attrs
    transformed_args = []
    for arg_name, arg_type in args:
        if arg_type in ["Symbol"]:
            continue
        if arg_type not in arg_type_dict:
            print(f"[NotImplemented] {arg_type} {arg_name}")
        transformed_args.append((arg_name, arg_type_dict[arg_type]))
    op_attrs[transformed_name] = transformed_args
    print(f"// {name}", file=attrs_header)
    if transformed_args:
        print(f"class {transformed_name}Attrs : public ir::AttrsNode<{transformed_name}Attrs> " + "{", file=attrs_header)
        print(" public:", file=attrs_header)
        for arg_name, arg_type in transformed_args:
            print(f"  {arg_type} {arg_name};", file=attrs_header)
        print("", file=attrs_header)
        print(f"  MX_V3_DECLARE_ATTRS({transformed_name}Attrs, \"mxnet.v3.attrs.{transformed_name}Attrs\")" + " {", file=attrs_header)
        for arg_name, _ in transformed_args:
            print(f"    MX_V3_ATTR_FIELD({arg_name});", file=attrs_header)
        print("  }", file=attrs_header)
        print("};", file=attrs_header)
        print(f"MX_V3_REGISTER_NODE_TYPE({transformed_name}Attrs);", file=attrs_cc)
    else:
        print(f"using {transformed_name}Attrs = ir::Attrs;", file=attrs_header)
        print(f"// Skip empty attribute {transformed_name}Attrs", file=attrs_cc)


def PrintAttrsHeader():
    attrs_header.write("""}  // namespace attrs
}  // namespace op
}  // namespace v3
}  // namespace mxnet
#endif""")


def PrintAttrsCC():
    attrs_cc.write("""}  // namespace
}  // namespace attrs
}  // namespace op
}  // namespace v3
}  // namespace mxnet
#endif
""")


def ParseAllOps():
    """
    MXNET_DLL int MXSymbolListAtomicSymbolCreators(mx_uint *out_size,
                                                   AtomicSymbolCreator **out_array);

    MXNET_DLL int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                              const char **name,
                                              const char **description,
                                              mx_uint *num_args,
                                              const char ***arg_names,
                                              const char ***arg_type_infos,
                                              const char ***arg_descriptions,
                                              const char **key_var_num_args);
    """
    cdll.libmxnet = cdll.LoadLibrary(sys.argv[1])
    ListOP = cdll.libmxnet.MXSymbolListAtomicSymbolCreators
    GetOpInfo = cdll.libmxnet.MXSymbolGetAtomicSymbolInfo
    ListOP.argtypes = [POINTER(c_int), POINTER(POINTER(c_void_p))]
    GetOpInfo.argtypes = [c_void_p,
                          POINTER(c_char_p),
                          POINTER(c_char_p),
                          POINTER(c_int),
                          POINTER(POINTER(c_char_p)),
                          POINTER(POINTER(c_char_p)),
                          POINTER(POINTER(c_char_p)),
                          POINTER(c_char_p),
                          POINTER(c_char_p)
                          ]

    nOps = c_int()
    opHandlers = POINTER(c_void_p)()
    r = ListOP(byref(nOps), byref(opHandlers))
    ret = ''
    ret2 = ''
    for i in range(0, nOps.value):
        handler = opHandlers[i]
        name = c_char_p()
        description = c_char_p()
        nArgs = c_int()
        argNames = POINTER(c_char_p)()
        argTypes = POINTER(c_char_p)()
        argDescs = POINTER(c_char_p)()
        varArgName = c_char_p()
        return_type = c_char_p()

        GetOpInfo(handler, byref(name), byref(description),
                  byref(nArgs), byref(argNames), byref(argTypes),
                  byref(argDescs), byref(varArgName), byref(return_type))

        name = name.value.decode('utf-8')

        if name.startswith("_backward_") or name.endswith("_backward"):
            # print("[Skip]", name)

            continue

        if name.startswith("_contrib_"):
            # print("[Skip]", name)

            continue

        if name.startswith("_image_"):
            # print("[Skip]", name)

            continue

        if name.endswith("_update"):
            # print("[Skip]", name)

            continue

        if name in blacklist:
            # print("[Skip]", name)

            continue
        args = []

        for j in range(0, nArgs.value):
            arg_name = argNames[j].decode('utf-8')
            arg_type = argTypes[j].decode('utf-8')
            arg_desc = argDescs[j].decode('utf-8')
            arg = Arg(name,
                      arg_name,
                      arg_type,
                      arg_desc)
            args.append(arg)
        literal_args = [(arg.name, arg.type) for arg in args]
        PrintAttrs(name, literal_args)
        op = Op(name, description.value.decode('utf-8'), args)
        ret = ret + op.GetOpDefinitionString(True) + "\n"
        ret2 = ret2 + op.GetOpDefinitionString(False) + "\n"
    PrintAttrsHeader()
    PrintAttrsCC()
    return ret + ret2


if __name__ == "__main__":
    #et = EnumType(typeName = 'MyET')
    # print(et.GetDefinitionString())
    # print(et.GetEnumStringArray())
    #arg = Arg()
    #print(arg.ConstructEnumTypeName('SoftmaxActivation', 'act_type'))
    # arg = Arg(opName = 'FullConnected', argName='act_type', \
    #    typeString="{'elu', 'leaky', 'prelu', 'rrelu'},optional, default='leaky'", \
    #    descString='Activation function to be applied.')
    # print(arg.isEnum)
    # print(arg.defaultString)
    #arg = Arg("fc", "alpha", "float, optional, default=0.0001", "alpha")
    #decl = "%s %s" % (arg.type, arg.name)
    # if arg.hasDefault:
    #    decl = decl + "=" + arg.defaultString
    # print(decl)

    temp_file_name = ""
    output_file = '../include/mxnet-cpp/op.h'
    try:
        # generate file header
        patternStr = ("/*!\n"
                      "*  Copyright (c) 2016 by Contributors\n"
                      "* \\file op.h\n"
                      "* \\brief definition of all the operators\n"
                      "* \\author Chuntao Hong, Xin Li\n"
                      "*/\n"
                      "\n"
                      "#ifndef MXNET_CPP_OP_H_\n"
                      "#define MXNET_CPP_OP_H_\n"
                      "\n"
                      "#include <string>\n"
                      "#include <vector>\n"
                      "#include \"mxnet-cpp/base.h\"\n"
                      "#include \"mxnet-cpp/shape.h\"\n"
                      "#include \"mxnet-cpp/op_util.h\"\n"
                      "#include \"mxnet-cpp/operator.h\"\n"
                      "#include \"dmlc/optional.h\"\n"
                      "#include \"nnvm/tuple.h\"\n"
                      "\n"
                      "namespace mxnet {\n"
                      "namespace cpp {\n"
                      "\n"
                      "%s"
                      "} //namespace cpp\n"
                      "} //namespace mxnet\n"
                      "#endif  // MXNET_CPP_OP_H_\n")

        # Generate a temporary file name
        tf = tempfile.NamedTemporaryFile()
        temp_file_name = tf.name
        tf.close()
        with codecs.open(temp_file_name, 'w', 'utf-8') as f:
            f.write(patternStr % ParseAllOps())
    except Exception as e:
        if (os.path.exists(output_file)):
            os.remove(output_file)

        if len(temp_file_name) > 0:
            os.remove(temp_file_name)
        raise(e)

    if os.path.exists(output_file):
        if not filecmp.cmp(temp_file_name, output_file):
            os.remove(output_file)
    # if not os.path.exists(output_file):
    #  shutil.move(temp_file_name, output_file)
