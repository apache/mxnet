/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package org.apache.mxnet.jnarator;

import org.apache.mxnet.jnarator.parser.CParser;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class FuncInfo {

    private String name;
    private DataType returnType;
    private List<Parameter> parameters = new ArrayList<>();

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public DataType getReturnType() {
        return returnType;
    }

    public void setReturnType(DataType returnType) {
        this.returnType = returnType;
    }

    public List<Parameter> getParameters() {
        return parameters;
    }

    public void setParameters(List<Parameter> parameters) {
        this.parameters = parameters;
    }

    public void addParameter(Parameter parameter) {
        parameters.add(parameter);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(returnType).append(' ').append(name).append('(');
        if (parameters != null) {
            boolean first = true;
            for (Parameter param : parameters) {
                if (first) {
                    first = false;
                } else {
                    sb.append(", ");
                }
                sb.append(param);
            }
        }

        sb.append(");");
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
        FuncInfo funcInfo = (FuncInfo) o;
        return name.equals(funcInfo.name) && parameters.equals(funcInfo.parameters);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(name);
    }

    static FuncInfo parse(CParser.DeclarationContext ctx) {
        FuncInfo info = new FuncInfo();

        List<CParser.DeclarationSpecifierContext> specs =
                ctx.declarationSpecifiers().declarationSpecifier();
        List<DataType> dataTypes = DataType.parseDataTypes(specs);
        info.setReturnType(dataTypes.get(0));
        if (dataTypes.size() > 1) {
            info.setName(dataTypes.get(1).getType());
        }

        CParser.InitDeclaratorContext init = ctx.initDeclaratorList().initDeclarator();
        CParser.DirectDeclaratorContext declarator = init.declarator().directDeclarator();

        CParser.DirectDeclaratorContext name = declarator.directDeclarator();
        if (info.getName() == null) {
            info.setName(name.getText());
            CParser.ParameterTypeListContext paramListCtx = declarator.parameterTypeList();
            if (paramListCtx != null) {
                Parameter.parseParams(info.getParameters(), paramListCtx);
            }
        } else {
            DataType dataType = new DataType();
            CParser.TypeSpecifierContext type = declarator.typeSpecifier();
            dataType.appendTypeName(type.getText());
            if (declarator.pointer() != null) {
                dataType.increasePointerCount();
            }
            Parameter param = new Parameter(dataType, name.getText());
            info.addParameter(param);
        }

        return info;
    }
}
