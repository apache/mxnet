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

package org.apache.mxnetexamples.javaapi.infer.predictor;

import org.apache.mxnetexamples.Util;
import org.apache.mxnetexamples.javaapi.infer.bert.BertQA;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Test on BERT QA model
 */
public class BertExampleTest {
    final static Logger logger = LoggerFactory.getLogger(BertExampleTest.class);
    private static String modelPathPrefix = "";
    private static String vocabPath = "";

    @BeforeClass
    public static void downloadFile() {
        logger.info("Downloading Bert QA Model");
        String tempDirPath = System.getProperty("java.io.tmpdir");
        logger.info("tempDirPath: %s".format(tempDirPath));

        String baseUrl = "https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/BertQA";
        Util.downloadUrl(baseUrl + "/static_bert_qa-symbol.json",
                tempDirPath + "/static_bert_qa/static_bert_qa-symbol.json", 3);
        Util.downloadUrl(baseUrl + "/static_bert_qa-0002.params",
                tempDirPath + "/static_bert_qa/static_bert_qa-0002.params", 3);
        Util.downloadUrl(baseUrl + "/vocab.json",
                tempDirPath + "/static_bert_qa/vocab.json", 3);
        modelPathPrefix = tempDirPath + File.separator + "static_bert_qa/static_bert_qa";
        vocabPath = tempDirPath + File.separator + "static_bert_qa/vocab.json";
    }

    @Test
    public void testBertQA() throws Exception{
        BertQA bert = new BertQA();
        String Q = "When did BBC Japan start broadcasting?";
        String A = "BBC Japan was a general entertainment Channel.\n" +
                " Which operated between December 2004 and April 2006.\n" +
                "It ceased operations after its Japanese distributor folded.";
        String[] args = new String[] {
                "--model-path-prefix", modelPathPrefix,
                "--model-vocab", vocabPath,
                "--model-epoch", "2",
                "--input-question", Q,
                "--input-answer", A,
                "--seq-length", "384"
        };
        bert.main(args);
    }
}
