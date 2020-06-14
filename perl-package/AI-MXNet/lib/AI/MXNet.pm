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

package AI::MXNet;
use v5.14.0;
use strict;
use warnings;
use AI::MXNet::NS 'global';
use AI::MXNet::Base;
use AI::MXNet::Callback 'callback';
use AI::MXNet::NDArray qw(nd ndarray);
use AI::MXNet::Context 'context';
use AI::MXNet::Symbol qw(sym symbol);
use AI::MXNet::Executor;
use AI::MXNet::Executor::Group;
use AI::MXNet::CudaModule;
use AI::MXNet::Random qw(rnd random);
use AI::MXNet::Initializer qw(init initializer);
use AI::MXNet::Optimizer qw(optimizer opt);
use AI::MXNet::KVStore 'kv';
use AI::MXNet::KVStoreServer;
use AI::MXNet::IO 'io';
use AI::MXNet::Metric 'metric';
use AI::MXNet::LRScheduler;
use AI::MXNet::Monitor 'mon';
use AI::MXNet::Profiler;
use AI::MXNet::Module::Base;
use AI::MXNet::Module qw(mod module);
use AI::MXNet::Module::Bucketing;
use AI::MXNet::RNN 'rnn';
use AI::MXNet::RunTime 'runtime';
use AI::MXNet::Visualization 'viz';
use AI::MXNet::RecordIO 'recordio';
use AI::MXNet::Image qw(img image);
use AI::MXNet::Contrib 'contrib';
use AI::MXNet::LinAlg 'linalg';
use AI::MXNet::CachedOp;
use AI::MXNet::AutoGrad 'autograd';
use AI::MXNet::Gluon 'gluon';
use AI::MXNet::NDArray::Sparse;
use AI::MXNet::Symbol::Sparse;
use AI::MXNet::Engine 'engine';
our $VERSION = '1.5';

sub cpu { AI::MXNet::Context->cpu($_[1]//0) }
sub cpu_pinned { AI::MXNet::Context->cpu_pinned($_[1]//0) }
sub gpu { AI::MXNet::Context->gpu($_[1]//0) }
sub name { __PACKAGE__ }
sub rtc { __PACKAGE__ }
sub Prefix { AI::MXNet::Symbol::Prefix->new(prefix => $_[1]) }
our $AttrScope = AI::MXNet::Symbol::AttrScope->new;
our $NameManager = AI::MXNet::Symbol::NameManager->new;
our $Context = AI::MXNet::Context->new(device_type => 'cpu', device_id => 0);

1;
__END__

=encoding UTF-8

=head1 NAME

AI::MXNet - Perl interface to MXNet machine learning library

=head1 SYNOPSIS

=head1 DESCRIPTION

    Perl interface to MXNet machine learning library.
    MXNet supports the Perl programming language.
    The MXNet Perl package brings flexible and efficient GPU computing and
    state-of-art deep learning to Perl.
    It enables you to write seamless tensor/matrix computation with multiple GPUs in Perl.
    It also lets you construct and customize the state-of-art deep learning models in Perl,
    and apply them to tasks, such as image classification and data science challenges.

    One important thing to internalize is that Perl interface is written to be as close as possible to the Python’s API,
    so most, if not all of Python’s documentation and examples should just work in Perl after making few changes
    in order to make the code a bit more Perlish. In nutshell just add $ sigils and replace . = \n with -> => ;
    and in 99% of cases that’s all that is needed there.
    In addition please refer to very detailed L<MXNet Python API Documentation|https://mxnet.apache.org/api/python/docs/tutorials/index.html>.

    AI::MXNet supports new imperative PyTorch like Gluon MXNet interface.
    Please get acquainted with this new interface at L<Dive into Deep Learning|https://www.d2l.ai/>.

    For specific Perl Gluon usage please refer to Perl examples and tests directories on github,
    but be assured that the Python and Perl usage are extremely close in order to make the use
    of the Python Gluon docs and examples as easy as possible.

    AI::MXNet is seamlessly glued with L<PDL|https://metacpan.org/pod/PDL>, the C++ level state can be easily initialized from PDL
    and the results can be transferred to PDL objects in order to allow you to use all the glory and power of the PDL!

=head1 BUGS AND INCOMPATIBILITIES

    Parity with Python interface is mostly achieved, few deprecated
    and not often used features left unported for now.

=head1 SEE ALSO

    L<https://mxnet.io/>
    L<https://github.com/dmlc/mxnet/tree/master/perl-package>
    L<Function::Parameters|https://metacpan.org/pod/Function::Parameters>, L<Mouse|https://metacpan.org/pod/Mouse>

=head1 AUTHOR

    Sergey Kolychev, <sergeykolychev.github@gmail.com>

=head1 COPYRIGHT & LICENSE

    This library is licensed under Apache 2.0 license L<https://www.apache.org/licenses/LICENSE-2.0>

=cut
