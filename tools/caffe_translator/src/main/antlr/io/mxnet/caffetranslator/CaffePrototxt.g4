/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
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

COMMENT :  '#' ~( '\r' | '\n' )* -> skip;
