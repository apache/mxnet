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
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

public final class AntlrUtils {

    private AntlrUtils() {}

    public static boolean isTypeDef(CParser.DeclarationSpecifiersContext specs) {
        if (specs.isEmpty()) {
            return false;
        }

        CParser.DeclarationSpecifierContext spec =
                (CParser.DeclarationSpecifierContext) specs.getChild(0);
        CParser.StorageClassSpecifierContext storage = spec.storageClassSpecifier();
        if (storage != null) {
            return storage.Typedef() != null;
        }
        return false;
    }

    public static String getTypeDefValue(CParser.DeclarationSpecifiersContext specs) {
        List<String> list = new ArrayList<>();
        for (int i = 1; i < specs.getChildCount(); ++i) {
            list.add(specs.getChild(i).getText());
        }
        return String.join(" ", list);
    }

    public static boolean isEnum(CParser.DeclarationSpecifiersContext specs) {
        if (specs.isEmpty()) {
            return false;
        }

        CParser.DeclarationSpecifierContext spec =
                (CParser.DeclarationSpecifierContext) specs.getChild(0);
        CParser.TypeSpecifierContext type = spec.typeSpecifier();
        if (type == null) {
            return false;
        }
        return type.enumSpecifier() != null;
    }

    public static boolean isStructOrUnion(CParser.DeclarationSpecifiersContext specs) {
        if (specs.isEmpty()) {
            return false;
        }

        CParser.DeclarationSpecifierContext spec =
                (CParser.DeclarationSpecifierContext) specs.getChild(0);
        CParser.TypeSpecifierContext type = spec.typeSpecifier();
        if (type == null) {
            return false;
        }
        return type.structOrUnionSpecifier() != null;
    }

    public static String getText(ParseTree tree) {
        StringBuilder sb = new StringBuilder();
        getText(sb, tree);
        return sb.toString();
    }

    private static void getText(StringBuilder sb, ParseTree tree) {
        if (tree instanceof TerminalNode) {
            sb.append("\"v\" : \"").append(tree.getText()).append('"');
            return;
        }
        sb.append('"');
        sb.append(tree.getClass().getSimpleName()).append("\" : {");
        for (int i = 0; i < tree.getChildCount(); i++) {
            getText(sb, tree.getChild(i));
            if (i < tree.getChildCount() - 1) {
                sb.append(',');
            }
        }
        sb.append('}');
    }

    public static String toCamelCase(String name) {
        String[] tokens = name.split("_");
        for (int i = 0; i < tokens.length; ++i) {
            char upper = Character.toUpperCase(tokens[i].charAt(0));
            tokens[i] = upper + tokens[i].substring(1); // NOPMD
        }
        return String.join("", tokens);
    }
}
