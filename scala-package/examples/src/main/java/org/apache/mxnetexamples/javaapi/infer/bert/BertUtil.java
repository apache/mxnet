package org.apache.mxnetexamples.javaapi.infer.bert;

import java.io.FileReader;
import java.util.*;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

public class BertUtil {

    private Map<String, Integer> token2idx;
    private List<String> idx2token;

    void parseJSON(String jsonFile) throws Exception {
        Gson gson = new Gson();
        token2idx = new HashMap<>();
        idx2token = new LinkedList<>();
        JsonObject jsonObject = gson.fromJson(new FileReader(jsonFile), JsonObject.class);
        JsonArray arr = jsonObject.getAsJsonArray("token_to_idx");
        for (JsonElement element : arr) {
            idx2token.add(element.getAsString());
        }
        JsonObject preMap = jsonObject.getAsJsonObject("idx_to_token");
        for (String key : preMap.keySet()) {
            token2idx.put(key, jsonObject.get(key).getAsInt());
        }
    }

    List<String> tokenizer(String input) {
        String[] step1 = input.split("[\n\r\t ]+");
        List<String> finalResult = new LinkedList<>();
        for (String item : step1) {
            if (item.length() != 0) {
                if (item.split("[.,?!]+").length > 1) {
                    finalResult.add(item.substring(0, item.length() - 1));
                    finalResult.add(item.substring(item.length() -1, item.length()));
                } else {
                    finalResult.add(item);
                }
            }
        }
        return finalResult;
    }

    <E> List<E> pad(List<E> tokens, E padItem, int num) {
        if (tokens.size() >= num) return tokens;
        List<E> padded = new LinkedList<>(tokens);
        for (int i = 0; i < num - tokens.size(); i++) {
            tokens.add(padItem);
        }
        return padded;
    }

    List<Integer> token2idx(List<String> tokens) {
        List<Integer> indexes = new ArrayList<>();
        for (String token : tokens) {
            if (token2idx.containsKey(token)) {
                indexes.add(token2idx.get(token));
            } else {
                indexes.add(token2idx.get("[UNK]"));
            }
        }
        return indexes;
    }

    <E> List<String> idx2token(List<E> indexes) {
        List<String> tokens = new ArrayList<>();
        for (E index : indexes) {
            tokens.add(idx2token.get((int) index));
        }
        return tokens;
    }
}
