package io.mxnet.caffetranslator;

import java.util.Collections;

public class Utils {
    public static String indent(String str, int level, boolean useSpaces, int numSpaces) {
        String prefix = null;
        if (!useSpaces) {
            prefix = String.join("", Collections.nCopies(level, "\t"));
        } else {
            String spaces = String.join("", Collections.nCopies(numSpaces, " "));
            prefix = String.join("", Collections.nCopies(level, spaces));
        }

        String indented = str.replaceAll("(?m)^", prefix);
        return indented;
    }
}
