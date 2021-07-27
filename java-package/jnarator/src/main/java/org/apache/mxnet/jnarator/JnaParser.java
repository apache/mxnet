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

import org.apache.mxnet.jnarator.parser.CBaseListener;
import org.apache.mxnet.jnarator.parser.CLexer;
import org.apache.mxnet.jnarator.parser.CParser;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class JnaParser {

    static final Logger logger = LoggerFactory.getLogger(Main.class);

    Map<String, List<TypeDefine>> structMap;
    Map<String, List<String>> enumMap;
    List<FuncInfo> functions;
    Map<String, TypeDefine> typedefMap;
    private Set<String> functionNames;

    public JnaParser() {
        structMap = new LinkedHashMap<>();
        enumMap = new LinkedHashMap<>();
        functions = new ArrayList<>();
        typedefMap = new LinkedHashMap<>();
        functionNames = new HashSet<>();
    }

    public void parse(String headerFile) {
        try {
            CLexer lexer = new CLexer(CharStreams.fromFileName(headerFile));
            CommonTokenStream tokens = new CommonTokenStream(lexer);
            CParser parser = new CParser(tokens);
            ParseTree tree = parser.compilationUnit();

            ParseTreeWalker walker = new ParseTreeWalker();
            CBaseListener listener =
                    new CBaseListener() {

                        /** {@inheritDoc} */
                        @Override
                        public void enterDeclaration(CParser.DeclarationContext ctx) {
                            CParser.DeclarationSpecifiersContext specs =
                                    ctx.declarationSpecifiers();
                            CParser.InitDeclaratorListContext init = ctx.initDeclaratorList();

                            if (AntlrUtils.isTypeDef(specs)) {
                                TypeDefine value = TypeDefine.parse(init, specs);
                                typedefMap.put(value.getDataType().getType(), value);
                            } else if (AntlrUtils.isStructOrUnion(specs)) {
                                CParser.DeclarationSpecifierContext spec =
                                        (CParser.DeclarationSpecifierContext) specs.getChild(0);
                                CParser.TypeSpecifierContext type = spec.typeSpecifier();
                                CParser.StructOrUnionSpecifierContext struct =
                                        type.structOrUnionSpecifier();
                                String name = struct.Identifier().getText();
                                List<TypeDefine> fields = new ArrayList<>();

                                CParser.StructDeclarationListContext list =
                                        struct.structDeclarationList();
                                parseStructFields(fields, list);

                                structMap.put(name, fields);
                            } else if (AntlrUtils.isEnum(specs)) {
                                CParser.DeclarationSpecifierContext spec =
                                        (CParser.DeclarationSpecifierContext) specs.getChild(0);
                                CParser.TypeSpecifierContext type = spec.typeSpecifier();
                                CParser.EnumSpecifierContext enumSpecifierContext =
                                        type.enumSpecifier();
                                String name = enumSpecifierContext.Identifier().getText();
                                List<String> fields = new ArrayList<>();
                                parseEnum(fields, ctx);
                                enumMap.put(name, fields);
                            } else {
                                FuncInfo info = FuncInfo.parse(ctx);
                                if (checkDuplicate(info)) {
                                    logger.warn("Duplicate function: {}.", info.getName());
                                } else {
                                    functions.add(info);
                                }
                            }
                        }
                    };
            walker.walk(listener, tree);
        } catch (IOException e) {
            logger.error("", e);
        }
    }

    void parseStructFields(List<TypeDefine> fields, ParseTree tree) {
        if (tree instanceof CParser.StructDeclarationContext) {
            CParser.StructDeclarationContext ctx = (CParser.StructDeclarationContext) tree;
            CParser.SpecifierQualifierListContext qualifierList = ctx.specifierQualifierList();
            DataType dataType = DataType.parse(qualifierList);

            TypeDefine typeDefine = new TypeDefine();
            fields.add(typeDefine);

            typeDefine.setDataType(dataType);

            CParser.StructDeclaratorListContext name = ctx.structDeclaratorList();
            if (name != null) {
                typeDefine.setCallBack(true);

                CParser.DirectDeclaratorContext dd =
                        name.structDeclarator().declarator().directDeclarator();
                CParser.DirectDeclaratorContext nameCtx =
                        dd.directDeclarator().declarator().directDeclarator();
                String fieldName = nameCtx.getText();
                typeDefine.setValue(fieldName);

                CParser.ParameterTypeListContext paramListCtx = dd.parameterTypeList();
                if (paramListCtx != null) {
                    Parameter.parseParams(typeDefine.getParameters(), paramListCtx);
                }
            } else {
                CParser.SpecifierQualifierListContext nameList =
                        qualifierList.specifierQualifierList();
                if (nameList.specifierQualifierList() != null) {
                    typeDefine.setValue(nameList.specifierQualifierList().getText());
                } else {
                    typeDefine.setValue(nameList.getText());
                }
            }
            return;
        }

        for (int i = 0; i < tree.getChildCount(); i++) {
            parseStructFields(fields, tree.getChild(i));
        }
    }

    void parseEnum(List<String> fields, ParseTree ctx) {
        if (ctx instanceof CParser.EnumerationConstantContext) {
            fields.add(ctx.getText());
            return;
        }

        for (int i = 0; i < ctx.getChildCount(); i++) {
            parseEnum(fields, ctx.getChild(i));
        }
    }

    public Map<String, List<TypeDefine>> getStructMap() {
        return structMap;
    }

    public Map<String, List<String>> getEnumMap() {
        return enumMap;
    }

    public List<FuncInfo> getFunctions() {
        return functions;
    }

    public Map<String, TypeDefine> getTypedefMap() {
        return typedefMap;
    }

    boolean checkDuplicate(FuncInfo function) {
        if (!functionNames.add(function.getName())) {
            for (FuncInfo info : functions) {
                if (function.equals(info)) {
                    return true;
                }
            }
        }
        return false;
    }
}
