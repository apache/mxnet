/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mxnet.jnarator;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;
import org.apache.mxnet.jnarator.parser.CParser;

public class DataType {

    private boolean isConst;
    private boolean functionPointer;
    private int pointerCount;
    private StringBuilder type = new StringBuilder(); // NOPMD

    public boolean isConst() {
        return isConst;
    }

    public void setConst() {
        isConst = true;
    }

    public boolean isFunctionPointer() {
        return functionPointer;
    }

    public void setFunctionPointer(boolean functionPointer) {
        this.functionPointer = functionPointer;
    }

    public int getPointerCount() {
        return pointerCount;
    }

    public void setPointerCount(int pointerCount) {
        this.pointerCount = pointerCount;
    }

    public void increasePointerCount() {
        ++pointerCount;
    }

    public String getType() {
        return type.toString();
    }

    public void setType(String typeName) {
        type.setLength(0);
        type.append(typeName);
    }

    public void appendTypeName(String name) {
        if (type.length() > 0) {
            type.append(' ');
        }
        type.append(name);
    }

    public String map(Map<String, TypeDefine> map, Set<String> structs) {
        String typeName = type.toString().trim();
        TypeDefine typeDefine = map.get(typeName);
        boolean isStruct = structs.contains(typeName);
        if (typeDefine != null && !typeDefine.isCallBack()) {
            typeName = typeDefine.getValue();

            String mapped = typeName.replaceAll("const ", "").replaceAll(" const", "");
            if (typeName.length() - mapped.length() > 0) {
                isConst = true;
            }
            typeName = mapped;
            mapped = typeName.replaceAll("\\*", "");
            int count = typeName.length() - mapped.length();
            pointerCount += count;
            typeName = mapped;
            setType(typeName);
        }

        if (pointerCount > 2) {
            return "PointerByReference";
        }

        typeName = baseTypeMapping(typeName);

        if (pointerCount == 2) {
            if (isConst && "char".equals(typeName)) {
                return "String[]";
            }
            return "PointerByReference";
        }

        if (pointerCount == 1) {
            switch (typeName) {
                case "byte":
                    return "ByteBuffer";
                case "NativeSize":
                    return "NativeSizeByReference";
                case "int":
                    if (isConst) {
                        return "int[]";
                    }
                    return "IntBuffer";
                case "long":
                    if (isConst) {
                        return "long[]";
                    }
                    return "LongBuffer";
                case "char":
                    if (isConst) {
                        return "String";
                    }
                    return "ByteBuffer";
                case "float":
                    return "FloatBuffer";
                case "void":
                    return "Pointer";
                default:
                    if (isStruct) {
                        return typeName + ".ByReference";
                    }
                    return "Pointer";
            }
        }
        return typeName;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        if (isConst) {
            sb.append("const ");
        }
        sb.append(type);
        if (pointerCount > 0) {
            sb.append(' ');
            for (int i = 0; i < pointerCount; ++i) {
                sb.append('*');
            }
        }
        return sb.toString();
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        DataType dataType = (DataType) o;
        return type.toString().equals(dataType.type.toString());
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(type);
    }

    static DataType parse(ParseTree tree) {
        DataType dataType = new DataType();
        parseTypeSpec(dataType, tree);
        return dataType;
    }

    static List<DataType> parseDataTypes(List<CParser.DeclarationSpecifierContext> list) {
        List<DataType> ret = new ArrayList<>();
        DataType dataType = new DataType();
        for (CParser.DeclarationSpecifierContext spec : list) {
            CParser.TypeQualifierContext qualifier = spec.typeQualifier();
            if (qualifier != null) {
                String qualifierName = qualifier.getText();
                if ("const".equals(qualifierName)) {
                    dataType.setConst();
                } else {
                    dataType.appendTypeName(qualifierName);
                }
                continue;
            }

            CParser.TypeSpecifierContext type = spec.typeSpecifier();
            parseTypeSpec(dataType, type);
            ret.add(dataType);
            dataType = new DataType();
        }

        return ret;
    }

    private static void parseTypeSpec(DataType dataType, ParseTree tree) {
        if (tree == null) {
            return;
        }

        if (tree instanceof CParser.StructOrUnionContext) {
            return;
        }
        if (tree instanceof CParser.TypedefNameContext) {
            if (dataType.getType().isEmpty()) {
                dataType.appendTypeName(tree.getText());
            }
            return;
        }

        if (tree instanceof TerminalNode) {
            String value = tree.getText();
            if ("const".equals(value)) {
                dataType.setConst();
            } else if ("*".equals(value)) {
                dataType.increasePointerCount();
            } else {
                dataType.appendTypeName(value);
            }
            return;
        }

        for (int i = 0; i < tree.getChildCount(); i++) {
            parseTypeSpec(dataType, tree.getChild(i));
        }
    }

    private static String baseTypeMapping(String type) {
        switch (type) {
            case "uint64_t":
            case "int64_t":
            case "long":
                return "long";
            case "uint32_t":
            case "unsigned int":
            case "unsigned":
            case "int":
                return "int";
            case "bool":
                return "byte";
            case "size_t":
                return "NativeSize";
            case "char":
            case "void":
            case "float":
            default:
                return type;
        }
    }
}
