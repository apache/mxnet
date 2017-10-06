package io.mxnet.caffetranslator;

public class ParserHelper {
    public String removeQuotes(String arg) {
        boolean doubleQuoteStr = (arg.startsWith("\"") && arg.endsWith("\""));
        boolean singleQuoteStr = (arg.startsWith("'") && arg.endsWith("'"));
        if ((singleQuoteStr | doubleQuoteStr) && arg.length() > 2) {
            arg = arg.substring(1, arg.length() - 1);
        }
        return arg;
    }
}
