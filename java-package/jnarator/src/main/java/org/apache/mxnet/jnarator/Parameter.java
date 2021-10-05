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

import java.util.List;
import java.util.Objects;
import org.antlr.v4.runtime.tree.ParseTree;
import org.apache.mxnet.jnarator.parser.CParser;

public class Parameter {

    private DataType type;
    private String name;

    public Parameter(DataType type, String name) {
        this.type = type;
        this.name = name;
    }

    public DataType getType() {
        return type;
    }

    public String getName() {
        return name;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return type.toString() + ' ' + name;
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
        Parameter parameter = (Parameter) o;
        return type.equals(parameter.type);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(type);
    }

    static void parseParams(List<Parameter> params, ParseTree ctx) {
        if (ctx instanceof CParser.ParameterDeclarationContext) {
            CParser.ParameterDeclarationContext declarationContext =
                    (CParser.ParameterDeclarationContext) ctx;
            CParser.DeclarationSpecifiersContext spec = declarationContext.declarationSpecifiers();
            DataType dataType;
            if (spec == null) {
                dataType = DataType.parse(declarationContext.declarationSpecifiers2());
            } else {
                dataType = DataType.parse(spec);
            }

            CParser.DeclaratorContext declarator = declarationContext.declarator();

            String name;
            if (declarator != null) {
                CParser.PointerContext pointer = declarator.pointer();
                if (pointer != null) {
                    dataType.increasePointerCount();
                }
                name = declarator.directDeclarator().getText();
            } else {
                name = "arg" + (params.size() + 1);
            }

            Parameter param = new Parameter(dataType, name);
            params.add(param);
            return;
        }
        for (int i = 0; i < ctx.getChildCount(); i++) {
            parseParams(params, ctx.getChild(i));
        }
    }
}
