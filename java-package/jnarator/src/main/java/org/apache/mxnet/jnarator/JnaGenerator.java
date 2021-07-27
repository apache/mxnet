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

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

public class JnaGenerator {

    private Path dir;
    private String packageName;
    private String libName;
    private String className;
    private Map<String, TypeDefine> typedefMap;
    private Set<String> structs;
    private Properties mapping;

    public JnaGenerator(
            String libName,
            String packageName,
            Map<String, TypeDefine> typedefMap,
            Set<String> structs,
            Properties mapping) {
        this.libName = libName;
        this.packageName = packageName;
        this.typedefMap = typedefMap;
        this.structs = structs;
        this.mapping = mapping;
    }

    public void init(String output) throws IOException {
        String[] tokens = packageName.split("\\.");
        dir = Paths.get(output, tokens);
        Files.createDirectories(dir);
        className = AntlrUtils.toCamelCase(libName) + "Library";
    }

    @SuppressWarnings("PMD.UseConcurrentHashMap")
    public void writeStructure(Map<String, List<TypeDefine>> structMap) throws IOException {
        for (Map.Entry<String, List<TypeDefine>> entry : structMap.entrySet()) {
            String name = entry.getKey();
            Path path = dir.resolve(name + ".java");
            try (BufferedWriter writer = Files.newBufferedWriter(path)) {
                writer.append("package ").append(packageName).append(";\n\n");

                Set<String> importSet = new HashSet<>();
                importSet.add("com.sun.jna.Pointer");
                importSet.add("com.sun.jna.Structure");
                importSet.add("java.util.List");

                Map<String, String> fieldNames = new LinkedHashMap<>();
                for (TypeDefine typeDefine : entry.getValue()) {
                    String fieldName = typeDefine.getValue();
                    String typeName;
                    if (typeDefine.isCallBack()) {
                        typeName = AntlrUtils.toCamelCase(fieldName) + "Callback";
                        importSet.add("com.sun.jna.Callback");
                        for (Parameter param : typeDefine.getParameters()) {
                            String type = param.getType().map(typedefMap, structs);
                            addImports(importSet, type);
                        }
                    } else {
                        typeName = typeDefine.getDataType().map(typedefMap, structs);
                        addImports(importSet, typeName);
                    }
                    fieldNames.put(fieldName, typeName);
                }

                int fieldCount = fieldNames.size();
                if (fieldCount < 2) {
                    importSet.add("java.util.Collections");
                } else {
                    importSet.add("java.util.Arrays");
                }

                List<String> imports = new ArrayList<>(importSet.size());
                imports.addAll(importSet);
                Collections.sort(imports);
                for (String imp : imports) {
                    writer.append("import ").append(imp).append(";\n");
                }

                writer.append("\npublic class ").append(name).append(" extends Structure {\n");
                if (fieldCount > 0) {
                    writer.write("\n");
                }
                for (Map.Entry<String, String> field : fieldNames.entrySet()) {
                    writer.append("    public ").append(field.getValue()).append(' ');
                    writer.append(field.getKey()).append(";\n");
                }

                writer.append("\n    public ").append(name).append("() {\n");
                writer.append("    }\n");
                writer.append("\n    public ").append(name).append("(Pointer peer) {\n");
                writer.append("        super(peer);\n");
                writer.append("    }\n");

                writer.append("\n    @Override\n");
                writer.append("    protected List<String> getFieldOrder() {\n");
                switch (fieldNames.size()) {
                    case 0:
                        writer.append("        return Collections.emptyList();\n");
                        break;
                    case 1:
                        writer.append("        return Collections.singletonList(");
                        String firstField = fieldNames.keySet().iterator().next();
                        writer.append('"').append(firstField).append("\");\n");
                        break;
                    default:
                        writer.append("        return Arrays.asList(");
                        boolean first = true;
                        for (String fieldName : fieldNames.keySet()) {
                            if (first) {
                                first = false;
                            } else {
                                writer.write(", ");
                            }
                            writer.append('"').append(fieldName).append('"');
                        }
                        writer.append(");\n");
                        break;
                }
                writer.append("    }\n");

                for (TypeDefine typeDefine : entry.getValue()) {
                    String fieldName = typeDefine.getValue();
                    String typeName = fieldNames.get(fieldName);
                    String getterName;
                    if (!typeDefine.isCallBack()) {
                        getterName = AntlrUtils.toCamelCase(fieldName);
                    } else {
                        getterName = typeName;
                    }

                    writer.append("\n    public void set").append(getterName).append('(');
                    writer.append(typeName).append(' ').append(fieldName).append(") {\n");
                    writer.append("        this.").append(fieldName).append(" = ");
                    writer.append(fieldName).append(";\n");
                    writer.append("    }\n");
                    writer.append("\n    public ").append(typeName).append(" get");
                    writer.append(getterName).append("() {\n");
                    writer.append("        return ").append(fieldName).append(";\n");
                    writer.append("    }\n");
                }

                writer.append("\n    public static final class ByReference extends ");
                writer.append(name).append(" implements Structure.ByReference {}\n");

                writer.append("\n    public static final class ByValue extends ");
                writer.append(name).append(" implements Structure.ByValue {}\n");

                for (TypeDefine typeDefine : entry.getValue()) {
                    if (typeDefine.isCallBack()) {
                        DataType dataType = typeDefine.getDataType();
                        String fieldName = typeDefine.getValue();

                        String callbackName = fieldNames.get(fieldName);
                        String returnType = mapping.getProperty(callbackName);
                        if (returnType == null) {
                            returnType = dataType.map(typedefMap, structs);
                        }

                        writer.append("\n    public interface ").append(callbackName);
                        writer.append(" extends Callback {\n");
                        writer.append("        ").append(returnType).append(" apply(");
                        writeParameters(writer, fieldName, typeDefine.getParameters());
                        writer.append(");\n");
                        writer.append("    }\n");
                    }
                }

                writer.append("}\n");
            }
        }
    }

    public void writeLibrary(Collection<FuncInfo> functions, Map<String, List<String>> enumMap)
            throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(dir.resolve(className + ".java"))) {
            writer.append("package ").append(packageName).append(";\n\n");

            writer.append("import com.sun.jna.Callback;\n");
            writer.append("import com.sun.jna.Library;\n");
            writer.append("import com.sun.jna.Pointer;\n");
            writer.append("import com.sun.jna.ptr.PointerByReference;\n");
            writer.append("import java.nio.ByteBuffer;\n");
            writer.append("import java.nio.FloatBuffer;\n");
            writer.append("import java.nio.IntBuffer;\n");
            writer.append("import java.nio.LongBuffer;\n");

            writer.append("\npublic interface ").append(className).append(" extends Library {\n\n");

            for (Map.Entry<String, List<String>> entry : enumMap.entrySet()) {
                String name = entry.getKey();
                writer.append("\n    enum ").append(name).append(" {\n");
                List<String> fields = entry.getValue();
                int count = 0;
                for (String field : fields) {
                    writer.append("        ").append(field);
                    if (++count < fields.size()) {
                        writer.append(',');
                    }
                    writer.append('\n');
                }
                writer.append("    }\n");
            }

            for (TypeDefine typeDefine : typedefMap.values()) {
                if (typeDefine.isCallBack()) {
                    String callbackName = typeDefine.getDataType().getType();
                    String returnType = mapping.getProperty(callbackName);
                    if (returnType == null) {
                        returnType = typeDefine.getValue();
                    }
                    writer.append("\n    interface ").append(callbackName);
                    writer.append(" extends Callback {\n");
                    writer.append("        ").append(returnType).append(" apply(");
                    writeParameters(writer, callbackName, typeDefine.getParameters());
                    writer.append(");\n");
                    writer.append("    }\n");
                }
            }

            for (FuncInfo info : functions) {
                writeFunction(writer, info);
            }
            writer.append("}\n");
        }
    }

    public void writeNativeSize() throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(dir.resolve("NativeSize.java"))) {
            writer.append("package ").append(packageName).append(";\n\n");
            writer.append("import com.sun.jna.IntegerType;\n");
            writer.append("import com.sun.jna.Native;\n\n");

            writer.append("public class NativeSize extends IntegerType {\n\n");
            writer.append("    private static final long serialVersionUID = 1L;\n\n");
            writer.append("    public static final int SIZE = Native.SIZE_T_SIZE;\n\n");
            writer.append("    public NativeSize() {\n");
            writer.append("        this(0);\n");
            writer.append("    }\n\n");
            writer.append("    public NativeSize(long value) {\n");
            writer.append("        super(SIZE, value);\n");
            writer.append("    }\n");
            writer.append("}\n");
        }

        Path path = dir.resolve("NativeSizeByReference.java");
        try (BufferedWriter writer = Files.newBufferedWriter(path)) {
            writer.append("package ").append(packageName).append(";\n\n");
            writer.append("import com.sun.jna.ptr.ByReference;\n\n");
            writer.append("public class NativeSizeByReference extends ByReference {\n\n");
            writer.append("    public NativeSizeByReference() {\n");
            writer.append("        this(new NativeSize(0));\n");
            writer.append("    }\n\n");
            writer.append("    public NativeSizeByReference(NativeSize value) {\n");
            writer.append("        super(NativeSize.SIZE);\n");
            writer.append("        setValue(value);\n");
            writer.append("    }\n\n");
            writer.append("    public void setValue(NativeSize value) {\n");
            writer.append("        if (NativeSize.SIZE == 4) {\n");
            writer.append("            getPointer().setInt(0, value.intValue());\n");
            writer.append("        } else if (NativeSize.SIZE == 8) {\n");
            writer.append("            getPointer().setLong(0, value.longValue());\n");
            writer.append("        } else {\n");
            writer.append(
                    "            throw new IllegalArgumentException(\"size_t has to be either 4 or 8 bytes.\");\n");
            writer.append("        }\n");
            writer.append("    }\n\n");
            writer.append("    public NativeSize getValue() {\n");
            writer.append("        if (NativeSize.SIZE == 4) {\n");
            writer.append("            return new NativeSize(getPointer().getInt(0));\n");
            writer.append("        } else if (NativeSize.SIZE == 8) {\n");
            writer.append("            return new NativeSize(getPointer().getLong(0));\n");
            writer.append("        } else {\n");
            writer.append(
                    "            throw new IllegalArgumentException(\"size_t has to be either 4 or 8 bytes.\");\n");
            writer.append("        }\n");
            writer.append("    }\n");
            writer.append("}\n");
        }
    }

    private void writeFunction(BufferedWriter writer, FuncInfo info) throws IOException {
        String funcName = info.getName();
        String returnType = mapping.getProperty(funcName);
        if (returnType == null) {
            returnType = info.getReturnType().map(typedefMap, structs);
        }
        writer.append("\n    ").append(returnType).append(' ');
        writer.append(funcName).append('(');
        writeParameters(writer, funcName, info.getParameters());
        writer.append(");\n");
    }

    private void writeParameters(BufferedWriter writer, String funcName, List<Parameter> parameters)
            throws IOException {
        if (parameters != null) {
            boolean first = true;
            for (Parameter param : parameters) {
                if (first) {
                    first = false;
                } else {
                    writer.append(", ");
                }
                String paramName = param.getName();
                String type = mapping.getProperty(funcName + '.' + paramName);
                if (type == null) {
                    type = param.getType().map(typedefMap, structs);
                }
                if (!"void".equals(type)) {
                    writer.append(type).append(' ');
                    writer.append(paramName);
                }
            }
        }
    }

    private static void addImports(Set<String> importSet, String typeName) {
        switch (typeName) {
            case "ByReference":
            case "ByteByReference":
            case "DoubleByReference":
            case "FloatByReference":
            case "IntByReference":
            case "LongByReference":
            case "NativeLongByReference":
            case "PointerByReference":
            case "ShortByReference":
                importSet.add("com.sun.jna.ptr." + typeName);
                break;
            case "ByteBuffer":
            case "DoubleBuffer":
            case "FloatBuffer":
            case "IntBuffer":
            case "LongBuffer":
            case "ShortBuffer":
                importSet.add("java.nio." + typeName);
                break;
            default:
                break;
        }
    }
}
