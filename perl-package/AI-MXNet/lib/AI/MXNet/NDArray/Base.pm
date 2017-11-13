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

package AI::MXNet::NDArray::Base;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::NDArray::Doc;
use Mouse;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet::NDArray::Base
=cut

=head1 DESCRIPTION

    This module provides a convenient interface to a C++ functions
    that work with NDArray.
    Essentially it loads them up during the lib startup into the Perl space.
=cut

my %function_meta;
method function_meta($code)
{
    return $function_meta{$code};
}

method function_meta_hash()
{
    return \%function_meta;
}

func _make_ndarray_function($handle, $func_name)
{
    my ($real_name, $desc, $arg_names,
        $arg_types, $arg_descs, $key_var_num_args,
        $ret_type) = @{ check_call(AI::MXNetCAPI::SymbolGetAtomicSymbolInfo($handle)) };
    $ret_type //= '';
    my $doc_str = build_doc($func_name,
                            $desc,
                            $arg_names,
                            $arg_types,
                            $arg_descs,
                            $key_var_num_args,
                            $ret_type
    );
    my %ndarguments;
    my @arguments;
    my %arguments = (out => 1, name => 1, ctx => 1, shape => 1);
    my $j = 0;
    for my $i (0..(@$arg_names-1))
    {
        if(not $arg_types->[$i] =~ /^(?:NDArray|Symbol|ndarray\-or\-symbol)/)
        {
            push @arguments, $arg_names->[$i];
            $arguments{ $arg_names->[$i] } = 1;
        }
        else
        {
            $ndarguments{ $arg_names->[$i] } = $j++;
        }
    }
    my $generic_ndarray_function = sub
    {
        my $class = shift;
        my (@args, %kwargs, %ndkwargs, @tmp);
        if(@_ and ref $_[-1] eq 'HASH')
        {
            %kwargs = %{ pop(@_) };
        }
        else
        {
            while(@_ >= 2 and not ref $_[-2])
            {
                if(exists $arguments{ $_[-2] })
                {
                    my $v = pop(@_);
                    my $k = pop(@_);
                    $kwargs{ $k } = $v;
                }
                elsif(exists $ndarguments{ $_[-2] })
                {
                    my $v = pop(@_);
                    my $k = pop(@_);
                    $ndkwargs{ $k } = $v;
                }
                else
                {
                    unshift(@tmp, pop(@_));
                    unshift(@tmp, pop(@_));
                }
            }
        }
        @args = (@_, @tmp);
        if(%ndkwargs)
        {
            for my $k (keys %ndkwargs)
            {
                $args[$ndarguments{$k}] = $ndkwargs{$k};
            }
        }
        my @ndargs;
        my @pos_args;
        for my $i (@args)
        {
            if(blessed($i) and $i->isa(__PACKAGE__))
            {
                push @ndargs, $i->handle;
            }
            else
            {
                push @pos_args, $i;
            }
            if(@pos_args > @arguments)
            {
                confess("Too many positional arguments");
            }
        }
        @kwargs{ @arguments[0..$#pos_args] } = @pos_args;
        my $original_output;
        my $output_vars;
        delete $kwargs{name};
        if(grep { $_ eq 'out' } keys %kwargs)
        {
            $output_vars = delete $kwargs{out};
            $original_output = $output_vars;
            unless(ref($output_vars) and ref($output_vars) eq 'ARRAY')
            {
                $output_vars = [$output_vars];
            }
        }
        else
        {
            $output_vars = [];
        }
        if(blessed($class) and $class->isa(__PACKAGE__) and not @{ $output_vars })
        {
            @ndargs = ($class->handle) if not @ndargs;
            $class = ref $class;
        }
        for my $key (keys %kwargs)
        {
            $kwargs{ $key } = "(" .join(", ", @{ $kwargs{ $key } }) .")"
                if ref $kwargs{ $key } eq 'ARRAY';
        }
        my $out = check_call(AI::MXNetCAPI::ImperativeInvoke(
                    $handle,
                    scalar(@ndargs),
                    \@ndargs,
                    [map { $_->handle } @$output_vars],
                    scalar(keys %kwargs),
                    \%kwargs)
        );
        return $original_output if $original_output;
        if(@$out == 1)
        {
            return $class->new(handle => $out->[0]);
        }
        else
        {
            return [map { $class->new(handle => $_) } @$out];
        }
    };
    $function_meta{ $generic_ndarray_function }{__name__} = $func_name;
    $function_meta{ $generic_ndarray_function }{__doc__} = $doc_str;
    return $generic_ndarray_function;
}

method _init_ndarray_module()
{
    my $op_names = check_call(AI::MXNetCAPI::ListAllOpNames());
    for my $name (@$op_names)
    {
        my $handle = check_call(AI::NNVMCAPI::GetOpHandle($name));
        my $function = _make_ndarray_function($handle, $name);
        {
            no strict 'refs';
            *{__PACKAGE__."::$name"} = $function;
        }
    }
}


__PACKAGE__->_init_ndarray_module;

1;
