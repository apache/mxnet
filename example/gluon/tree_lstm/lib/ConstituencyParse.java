/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import edu.stanford.nlp.process.WordTokenFactory;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.TreeBinarizer;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.StringReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.HashMap;
import java.util.Properties;
import java.util.Scanner;

public class ConstituencyParse {

  private boolean tokenize;
  private BufferedWriter tokWriter, parentWriter;
  private LexicalizedParser parser;
  private TreeBinarizer binarizer;
  private CollapseUnaryTransformer transformer;
  private GrammaticalStructureFactory gsf;

  private static final String PCFG_PATH = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";

  public ConstituencyParse(String tokPath, String parentPath, boolean tokenize) throws IOException {
    this.tokenize = tokenize;
    if (tokPath != null) {
      tokWriter = new BufferedWriter(new FileWriter(tokPath));
    }
    parentWriter = new BufferedWriter(new FileWriter(parentPath));
    parser = LexicalizedParser.loadModel(PCFG_PATH);
    binarizer = TreeBinarizer.simpleTreeBinarizer(
      parser.getTLPParams().headFinder(), parser.treebankLanguagePack());
    transformer = new CollapseUnaryTransformer();

    // set up to produce dependency representations from constituency trees
    TreebankLanguagePack tlp = new PennTreebankLanguagePack();
    gsf = tlp.grammaticalStructureFactory();
  }

  public List<HasWord> sentenceToTokens(String line) {
    List<HasWord> tokens = new ArrayList<>();
    if (tokenize) {
      PTBTokenizer<Word> tokenizer = new PTBTokenizer(new StringReader(line), new WordTokenFactory(), "");
      for (Word label; tokenizer.hasNext(); ) {
        tokens.add(tokenizer.next());
      }
    } else {
      for (String word : line.split(" ")) {
        tokens.add(new Word(word));
      }
    }

    return tokens;
  }

  public Tree parse(List<HasWord> tokens) {
    Tree tree = parser.apply(tokens);
    return tree;
  }

  public int[] constTreeParents(Tree tree) {
    Tree binarized = binarizer.transformTree(tree);
    Tree collapsedUnary = transformer.transformTree(binarized);
    Trees.convertToCoreLabels(collapsedUnary);
    collapsedUnary.indexSpans();
    List<Tree> leaves = collapsedUnary.getLeaves();
    int size = collapsedUnary.size() - leaves.size();
    int[] parents = new int[size];
    HashMap<Integer, Integer> index = new HashMap<Integer, Integer>();

    int idx = leaves.size();
    int leafIdx = 0;
    for (Tree leaf : leaves) {
      Tree cur = leaf.parent(collapsedUnary); // go to preterminal
      int curIdx = leafIdx++;
      boolean done = false;
      while (!done) {
        Tree parent = cur.parent(collapsedUnary);
        if (parent == null) {
          parents[curIdx] = 0;
          break;
        }

        int parentIdx;
        int parentNumber = parent.nodeNumber(collapsedUnary);
        if (!index.containsKey(parentNumber)) {
          parentIdx = idx++;
          index.put(parentNumber, parentIdx);
        } else {
          parentIdx = index.get(parentNumber);
          done = true;
        }

        parents[curIdx] = parentIdx + 1;
        cur = parent;
        curIdx = parentIdx;
      }
    }

    return parents;
  }

  // convert constituency parse to a dependency representation and return the
  // parent pointer representation of the tree
  public int[] depTreeParents(Tree tree, List<HasWord> tokens) {
    GrammaticalStructure gs = gsf.newGrammaticalStructure(tree);
    Collection<TypedDependency> tdl = gs.typedDependencies();
    int len = tokens.size();
    int[] parents = new int[len];
    for (int i = 0; i < len; i++) {
      // if a node has a parent of -1 at the end of parsing, then the node
      // has no parent.
      parents[i] = -1;
    }

    for (TypedDependency td : tdl) {
      // let root have index 0
      int child = td.dep().index();
      int parent = td.gov().index();
      parents[child - 1] = parent;
    }

    return parents;
  }

  public void printTokens(List<HasWord> tokens) throws IOException {
    int len = tokens.size();
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < len - 1; i++) {
      if (tokenize) {
        sb.append(PTBTokenizer.ptbToken2Text(tokens.get(i).word()));
      } else {
        sb.append(tokens.get(i).word());
      }
      sb.append(' ');
    }

    if (tokenize) {
      sb.append(PTBTokenizer.ptbToken2Text(tokens.get(len - 1).word()));
    } else {
      sb.append(tokens.get(len - 1).word());
    }

    sb.append('\n');
    tokWriter.write(sb.toString());
  }

  public void printParents(int[] parents) throws IOException {
    StringBuilder sb = new StringBuilder();
    int size = parents.length;
    for (int i = 0; i < size - 1; i++) {
      sb.append(parents[i]);
      sb.append(' ');
    }
    sb.append(parents[size - 1]);
    sb.append('\n');
    parentWriter.write(sb.toString());
  }

  public void close() throws IOException {
    if (tokWriter != null) tokWriter.close();
    parentWriter.close();
  }

  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    if (!props.containsKey("parentpath")) {
      System.err.println(
        "usage: java ConstituencyParse -deps - -tokenize - -tokpath <tokpath> -parentpath <parentpath>");
      System.exit(1);
    }

    // whether to tokenize input sentences
    boolean tokenize = false;
    if (props.containsKey("tokenize")) {
      tokenize = true;
    }

    // whether to produce dependency trees from the constituency parse
    boolean deps = false;
    if (props.containsKey("deps")) {
      deps = true;
    }

    String tokPath = props.containsKey("tokpath") ? props.getProperty("tokpath") : null;
    String parentPath = props.getProperty("parentpath");
    ConstituencyParse processor = new ConstituencyParse(tokPath, parentPath, tokenize);

    Scanner stdin = new Scanner(System.in);
    int count = 0;
    long start = System.currentTimeMillis();
    while (stdin.hasNextLine()) {
      String line = stdin.nextLine();
      List<HasWord> tokens = processor.sentenceToTokens(line);
      Tree parse = processor.parse(tokens);

      // produce parent pointer representation
      int[] parents = deps ? processor.depTreeParents(parse, tokens)
                           : processor.constTreeParents(parse);

      // print
      if (tokPath != null) {
        processor.printTokens(tokens);
      }
      processor.printParents(parents);

      count++;
      if (count % 1000 == 0) {
        double elapsed = (System.currentTimeMillis() - start) / 1000.0;
        System.err.printf("Parsed %d lines (%.2fs)\n", count, elapsed);
      }
    }

    long totalTimeMillis = System.currentTimeMillis() - start;
    System.err.printf("Done: %d lines in %.2fs (%.1fms per line)\n",
      count, totalTimeMillis / 1000.0, totalTimeMillis / (double) count);
    processor.close();
  }
}
