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

/*!
 * \file CaffePrototxt.g4
 * \brief Grammar to parse Caffe prototxt
 */

grammar CaffePrototxt;

@header {
package io.mxnet.caffetranslator;
}


prototxt: name layer+;

solver: pair+;

name: ID COLON STRING;

layer: ID object;

pair: ID COLON? value;

value: object                   #valueObject
     | (STRING | NUMBER | ID)   #valueLeaf
     ;

object: LPAREN pair+ RPAREN;

LPAREN: '{';

RPAREN: '}';

COLON: ':';

NUMBER : '-'? ('.' DIGIT+ | DIGIT+ ('.' DIGIT*)? ) Exponent?;
fragment
DIGIT : [0-9] ;
fragment
Exponent : ('e'|'E') ('+'|'-')? ('0'..'9')+ ;

ID: LETTER (LETTER|DIGIT)*;

fragment
LETTER      :   [a-zA-Z\u0080-\u00FF_] ;

STRING      :   '"' ('\\"'|.)*? '"'
            |   '\'' ('\\\''|.)*? '\'' ;

WS  :   [ \t]+ -> channel(HIDDEN) ;

NL  :   [\n\r]+ -> channel(HIDDEN) ;

COMMENT :  '#' ~( '\r' | '\n' )* {!getText().startsWith("#CaffeToMXNet")}? -> skip;

CAFFE2MXNET: '#CaffeToMXNet' -> skip;
