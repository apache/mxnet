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
    my @arguments;
    for my $i (0..(@$arg_names-1))
    {
        if(not $arg_types->[$i] =~ /^(?:NDArray|Symbol|ndarray\-or\-symbol)/)
        {
            push @arguments, $arg_names->[$i];
        }
    }
    my $generic_ndarray_function = sub
    {
        my $class = shift;
        my (@args, %kwargs);
        if(@_ and ref $_[-1] eq 'HASH')
        {
            %kwargs = %{ pop(@_) };
        }
        @args = @_;
        if(ref $class)
        {
            @args = ($class) if not @args;
            $class = ref $class;
        }
        my @ndargs;
        my @pos_args;
        for my $i (@args)
        {
            if(blessed($i) and $i->isa($class))
            {
                push @ndargs, $i->handle;
            }
            else
            {
                push @pos_args, $i;
            }
            if(@pos_args > @arguments)
            {
                die "Too many positional arguments";
            }
        }
        @kwargs{ @arguments[0..$#pos_args] } = @pos_args;
        my $original_output;
        my $output_vars;
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
