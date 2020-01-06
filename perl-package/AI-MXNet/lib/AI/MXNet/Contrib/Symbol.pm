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

package AI::MXNet::Contrib::Symbol;
use strict;
use warnings;
use AI::MXNet::NS;
use parent 'AI::MXNet::AutoLoad';
sub config { ('contrib', 'AI::MXNet::Symbol') }

=head1 NAME

    AI::MXNet::Contrib - An interface to experimental symbol operators defined in C++ space.
=cut

=head1 SYNOPSIS

    my $embed;
    if($sparse_embedding)
    {
        my $embed_weight = mx->sym->Variable('embed_weight', stype=>'row_sparse');
        $embed = mx->sym->contrib->SparseEmbedding(
            data=>$data, input_dim=>$num_words,
            weight=>$embed_weight, output_dim=>$num_embed,
            name=>'embed'
        );
    }
    else
    {
        $embed = mx->sym->Embedding(
            data=>$data, input_dim=>$num_words,
            output_dim=>$num_embed, name=>'embed'
        );
    }
=cut

1;
