.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.  See the License for the
   specific language governing permissions and limitations
   under the License.

Text Tutorials
==============

These tutorials will help you learn how to create and use models that work with text and other natural language processing tasks.

Word Embedding
--------------

.. container:: cards

   .. card::
      :title: Pre-trained Word Embeddings
      :link: https://gluon-nlp.mxnet.io/examples/word_embedding/word_embedding.html

      Basics on how to use word embedding with vocab in GluonNLP and apply it on word similarity and analogy problems.

   .. card::
      :title: Word Embeddings Training and Evaluation
      :link: https://gluon-nlp.mxnet.io/examples/word_embedding/word_embedding_training.html

      Learn how to train fastText and word2vec embeddings on your own dataset, and determine embedding quality through intrinsic evaluation.

Language Model
--------------


.. container:: cards

   .. card::
      :title: LSTM-based Language Models
      :link: https://gluon-nlp.mxnet.io/examples/language_model/language_model.html

      Learn what a language model is, what it can do, and how to train a word-level language model with truncated back-propagation-through-time (BPTT).

Machine Translation
-------------------

.. container:: cards

   .. card::
      :title: Google Neural Machine Translation
      :link: https://gluon-nlp.mxnet.io/examples/machine_translation/gnmt.html

      Learn how to train Google Neural Machine Translation, a seq2seq with attention model.

   .. card::
      :title: Machine Translation with Transformer
      :link: https://gluon-nlp.mxnet.io/examples/machine_translation/transformer.html

      Learn how to use a pre-trained transformer translation model for English to German translation.

Sentence Embedding
---------------------

.. container:: cards

   .. card::
      :title: ELMo: Deep Contextualized Word Representations
      :link: https://gluon-nlp.mxnet.io/examples/sentence_embedding/elmo_sentence_representation.html

      See how to use GluonNLP’s model API to automatically download the pre-trained ELMo model from NAACL2018 best paper, and extract features with it.

   .. card::
      :title: A Structured Self-attentive Sentence Embedding
      :link: https://gluon-nlp.mxnet.io/examples/sentence_embedding/self_attentive_sentence_embedding.html

      See how to use GluonNLP to build more advanced model structure for extracting sentence embeddings to predict Yelp review rating.

   .. card::
      :title: BERT: Bidirectional Encoder Representations from Transformers
      :link: https://gluon-nlp.mxnet.io/examples/sentence_embedding/bert.html

      See how to use GluonNLP to fine-tune a sentence pair classification model with pre-trained BERT parameters.

Sentiment Analysis
------------------

.. container:: cards

   .. card::
      :title: Sentiment Analysis by Fine-tuning Word Language Model
      :link: https://gluon-nlp.mxnet.io/examples/sentiment_analysis/sentiment_analysis.html

      See how to fine-tune a pre-trained language model to perform sentiment analysis on movie reviews.

Sequence Sampling
-----------------

.. container:: cards

   .. card::
      :title: Sequence Generation with Sampling and Beam Search
      :link: https://gluon-nlp.mxnet.io/examples/sequence_sampling/sequence_sampling.html

      Learn how to generate sentence from pre-trained language model through sampling and beam search. 

.. toctree::
   :hidden:
   :maxdepth: 1
   :glob:

   *
