# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

use strict;
use warnings;

use 5.014000;

use ExtUtils::MakeMaker 6.30;



my %WriteMakefileArgs = (
  "ABSTRACT" => "Perl interface to MXNet machine learning library",
  "AUTHOR" => "Sergey Kolychev <sergeykolychev.github\@gmail.com>",
  "BUILD_REQUIRES" => {},
  "CONFIGURE_REQUIRES" => {
    "ExtUtils::MakeMaker" => "6.30"
  },
  "DISTNAME" => "AI-MXNet",
  "EXE_FILES" => [],
  "LICENSE" => "apache_2_0",
  "NAME" => "AI::MXNet",
  "PREREQ_PM" => {
    "AI::MXNetCAPI" => "1.1",
    "AI::NNVMCAPI" => "1.1",
    "Function::Parameters" => "1.0705",
    "Hash::Ordered" => "0.012",
    "Mouse" => "v2.1.0",
    "PDL" => "2.007",
    "GraphViz" => "2.14"
  },
  "TEST_REQUIRES" => {},
  "VERSION" => "1.1",
  "test" => {
    "TESTS" => "t/*.t"
  }
);


my %FallbackPrereqs = (
  "AI::MXNetCAPI" => "1.1",
  "AI::NNVMCAPI" => "1.1",
  "Function::Parameters" => "1.0705",
  "Hash::Ordered" => "0.012",
  "Mouse" => "v2.1.0",
  "PDL" => "2.007",
  "GraphViz" => "2.14"
);


unless ( eval { ExtUtils::MakeMaker->VERSION(6.63_03) } ) {
  delete $WriteMakefileArgs{TEST_REQUIRES};
  delete $WriteMakefileArgs{BUILD_REQUIRES};
  $WriteMakefileArgs{PREREQ_PM} = \%FallbackPrereqs;
}

delete $WriteMakefileArgs{CONFIGURE_REQUIRES}
  unless eval { ExtUtils::MakeMaker->VERSION(6.52) };

WriteMakefile(%WriteMakefileArgs);



