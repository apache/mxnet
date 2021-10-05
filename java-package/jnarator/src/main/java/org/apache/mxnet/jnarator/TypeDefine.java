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
import org.apache.mxnet.jnarator.parser.CParser;

public class TypeDefine {

    private DataType dataType;
    private boolean callBack;
    private String value;
    private List<Parameter> parameters = new ArrayList<>();

    public DataType getDataType() {
        return dataType;
    }

    public void setDataType(DataType dataType) {
        this.dataType = dataType;
    }

    public boolean isCallBack() {
        return callBack;
    }

    public void setCallBack(boolean callBack) {
        this.callBack = callBack;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public List<Parameter> getParameters() {
        return parameters;
    }

    static TypeDefine parse(
            CParser.InitDeclaratorListContext init, CParser.DeclarationSpecifiersContext specs) {
        TypeDefine typeDefine = new TypeDefine();
        DataType dataType = new DataType();
        typeDefine.setDataType(dataType);

        CParser.DirectDeclaratorContext ctx = init.initDeclarator().declarator().directDeclarator();
        CParser.DirectDeclaratorContext callback = ctx.directDeclarator();
        if (callback == null) {
            dataType.setType(ctx.getText());
        } else {
            typeDefine.setCallBack(true);
            dataType.setType(callback.declarator().directDeclarator().getText());
            CParser.ParameterTypeListContext paramListCtx = ctx.parameterTypeList();
            List<Parameter> parameters = typeDefine.getParameters();
            Parameter.parseParams(parameters, paramListCtx);
        }

        List<String> list = new ArrayList<>();
        for (int i = 1; i < specs.getChildCount(); ++i) {
            list.add(specs.getChild(i).getText());
        }

        typeDefine.setValue(String.join(" ", list));
        return typeDefine;
    }
}
