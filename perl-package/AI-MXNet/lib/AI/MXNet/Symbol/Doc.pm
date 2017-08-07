package AI::MXNet::Symbol::Doc;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;
use Exporter;
use base qw(Exporter);
@AI::MXNet::Symbol::Doc::EXPORT = qw/build_doc/;

method get_output_shape(AI::MXNet::Symbol $sym, %input_shapes)
{
    my $s_outputs = $sym->infer_shape(%input_shapes);
    my %ret;
    @ret{ @{ $sym->list_outputs() } } = @$s_outputs;
    return bless \%ret, 'AI::MXNet::Util::Printable';
}

func build_doc(
                    Str $func_name,
                    Str $desc,
                    ArrayRef[Str] $arg_names,
                    ArrayRef[Str] $arg_types,
                    ArrayRef[Str] $arg_desc,
                    Str $key_var_num_args=,
                    Str $ret_type=
)
{
    my $param_str = build_param_doc($arg_names, $arg_types, $arg_desc);
    if($key_var_num_args)
    {
        $desc .= "\nThis function support variable length of positional input."
    }
    my $doc_str = sprintf("%s\n\n" .
               "%s\n" .
               "name : string, optional.\n" .
               "    Name of the resulting symbol.\n\n" .
               "Returns\n" .
               "-------\n" .
               "symbol: Symbol\n" .
               "    The result symbol.", $desc, $param_str);
    return $doc_str;
}

1;
