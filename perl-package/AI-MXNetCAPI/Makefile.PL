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

use ExtUtils::MakeMaker;
use DynaLoader;
`swig -noproxy -c++ -perl mxnet.i`;
unlink "MXNetCAPI.pm";
my @tmp = split(/ /, $ExtUtils::MakeMaker::Config{lddlflags});
my @lddlflags;
while(my $flag = shift(@tmp))
{
   if($flag eq '-arch')
   {
      my $arch = shift(@tmp);
      if($arch eq 'i386')
      {
         next;
      }
      else
      {
         push @lddlflags, ($flag, $arch);
      }
   }
   else
   {
      push @lddlflags, $flag;
   }
}
WriteMakefile(
    NAME            => 'AI::MXNetCAPI',
    LICENSE         => 'apache_2_0',
    AUTHOR          => 'Sergey Kolychev <sergeykolychev.github@gmail.com>',
    VERSION_FROM    => 'lib/AI/MXNetCAPI.pm',
    ABSTRACT_FROM   => 'lib/AI/MXNetCAPI.pm',
    LIBS           => ['-L../../lib -lmxnet'],
    INC            => '-I../../include/mxnet',
    OBJECT         => 'mxnet_wrap.o',
    LDDLFLAGS      => join(' ', @lddlflags),
    PREREQ_PM      => {
        # prereqs
        # build/test prereqs
        'Test::More'   => 0,
    },
    PL_FILES        => {},
);
