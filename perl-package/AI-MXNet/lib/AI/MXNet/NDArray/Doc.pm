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

package AI::MXNet::NDArray::Doc;
use strict;
use warnings;
use AI::MXNet::Base;
use Exporter;
use base qw(Exporter);
@AI::MXNet::NDArray::Doc::EXPORT = qw(build_doc);

=head2

    Build docstring for imperative functions.
=cut

sub build_doc
{
    my ($func_name,
        $desc,
        $arg_names,
        $arg_types,
        $arg_desc,
        $key_var_num_args,
        $ret_type) = @_;
    my $param_str = build_param_doc($arg_names, $arg_types, $arg_desc);
    if($key_var_num_args)
    {
        $desc .= "\nThis function support variable length of positional input."
    }
    my $doc_str = sprintf("%s\n\n" .
               "%s\n" .
               "out : NDArray, optional\n" .
               "    The output NDArray to hold the result.\n\n".
               "Returns\n" .
               "-------\n" .
               "out : NDArray or list of NDArray\n" .
               "    The output of this function.", $desc, $param_str);
    return $doc_str
}

1;
