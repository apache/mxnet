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
