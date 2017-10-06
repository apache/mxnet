package io.mxnet.caffetranslator.misc;

import io.mxnet.caffetranslator.CaffePrototxtLexer;
import io.mxnet.caffetranslator.CaffePrototxtParser;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;

import java.io.File;
import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

public class CollectStats {

    public static void main(String arsg[]) {
        String filePath = "path";

        CharStream cs = null;
        try {
            FileInputStream fis = new FileInputStream(new File(filePath));
            cs = CharStreams.fromStream(fis, StandardCharsets.UTF_8);
        } catch (Exception e) {
            e.printStackTrace();
        }

        CaffePrototxtLexer lexer = new CaffePrototxtLexer(cs);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        CaffePrototxtParser parser = new CaffePrototxtParser(tokens);

        StatsListener statsListener = new StatsListener();
        parser.addParseListener(statsListener);
        parser.prototxt();

        Map<String, Set<String>> attrMap = statsListener.getAttrMap();

        Iterator it = attrMap.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<String, Set<String>> pair = (Map.Entry) it.next();
            System.out.println(pair.getKey() + ":");
            for (String value : pair.getValue()) {
                System.out.println("    " + value);
            }
        }
    }

}
