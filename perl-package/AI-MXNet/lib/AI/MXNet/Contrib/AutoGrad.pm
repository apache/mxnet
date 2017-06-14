package AI::MXNet::Contrib::AutoGrad;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;
use Scalar::Util qw(blessed);

=head1 NAME

    AI::MXNet::AutoGrad - Autograd for NDArray.
=cut

=head2 set_is_training

    Set status to training/not training. When training, graph will be constructed
    for gradient computation. Operators will also run with ctx.is_train=True. For example,
    Dropout will drop inputs randomly when is_train=True while simply passing through
    if is_train=False.

    Parameters
    ----------
    is_train: bool

    Returns
    -------
    previous state before this set.
=cut


method set_is_training(Bool $is_train)
{
    my $prev = scalar(check_call(AI::MXNetCAPI::AutogradSetIsTraining($is_train ? 1 : 0)));
    return $prev ? 1 : 0
}

=head2 mark_variables

    Mark AI::MXNet::NDArrays as variables to compute gradient for autograd.

    Parameters
    ----------
    variables: array ref of AI::MXNet::NDArrays
    gradients: array ref of AI::MXNet::NDArrays
    grad_reqs: array ref of strings
=cut

method mark_variables(
    ArrayRef[AI::MXNet::NDArray]  $variables,
    ArrayRef[AI::MXNet::NDArray]  $gradients,
    GradReq|ArrayRef[GradReq]     $grad_reqs='write'
)
{
    my @variable_handles = map { $_->handle } @{ $variables };
    my @gradient_handles = map { $_->handle } @{ $gradients };
    my @grad_reqs;
    if(not ref $grad_reqs)
    {
        @grad_reqs = (GRAD_REQ_MAP->{ $grad_reqs }) x scalar(@variable_handles);
    }
    else
    {
        @grad_reqs = map { GRAD_REQ_MAP->{ $_ } } @{ $grad_reqs };
    }
    check_call(
        AI::MXNetCAPI::AutogradMarkVariables(
            scalar(@variable_handles),
            \@variable_handles,
            \@grad_reqs,
            \@gradient_handles
        )
    );
}

=head2 backward

     Compute the gradients of outputs w.r.t variables.

     Parameters
     ----------
     outputs: array ref of NDArray
     out_grads: array ref of NDArray or undef
     retain_graph: bool, defaults to false
=cut


method backward(
    ArrayRef[AI::MXNet::NDArray] $outputs,
    Maybe[ArrayRef[AI::MXNet::NDArray|Undef]] $out_grads=,
    Bool $retain_graph=0
)
{
    my @output_handles = map { $_->handle } @{ $outputs };
    if(not defined $out_grads)
    {
        check_call(
            AI::MXNetCAPI::AutogradBackward(
                scalar(@output_handles),
                \@output_handles,
                [],
                $retain_graph
            )
        );
        return;
    }

    my @ograd_handles;
    for my $arr (@$out_grads)
    {
        push @ograd_handles, (defined $arr ? $arr->handle : undef);
    }
    assert(
        (@ograd_handles == @output_handles),
        "outputs and out_grads must have the same length"
    );

    check_call(
        AI::MXNetCAPI::AutogradBackward(
            scalar(@output_handles),
            \@output_handles,
            \@ograd_handles,
            $retain_graph
        )
    );
}

=head2 compute_gradient

    Compute the gradients of outputs w.r.t variables.

    Parameters
    ----------
    outputs: array ref of NDArray

    Returns
    -------
    gradients: array ref of NDArray
=cut


method compute_gradient(ArrayRef[AI::MXNet::NDArray] $outputs)
{
    __PACKAGE__->backward($outputs);
}

=head2 grad_and_loss

    Return function that computes both gradient of arguments and loss value.

    Parameters
    ----------
    func: a perl sub
        The forward (loss) function.
    argnum: an int or a array ref of int
        The index of argument to calculate gradient for.

    Returns
    -------
    grad_and_loss_func: a perl sub
        A function that would compute both the gradient of arguments and loss value.
=cut

method grad_and_loss(CodeRef $func, Maybe[Int|ArrayRef[Int]] $argnum=)
{
    return sub {
        my @args = @_;
        my @variables = @_;
        if(defined $argnum)
        {
            my @argnum = ref $argnum ? @$argnum : ($argnum);
            @variables = map { $_[$_] } @argnum;
        }
        map {
            assert(
                (blessed($_) and $_->isa('AI::MXNet::NDArray')),
                "type of autograd input should NDArray")
        } @variables;
        my @grads = map { $_->zeros_like } @variables;
        __PACKAGE__->mark_variables(\@variables, \@grads);
        my $prev = __PACKAGE__->set_is_training(1);
        my $outputs = $func->(@args);
        __PACKAGE__->set_is_training(0) unless $prev;
        __PACKAGE__->compute_gradient(ref $outputs eq 'ARRAY' ? $outputs : [$outputs]);
        return (\@grads, $outputs);
    };
}

=head2 grad

    Return function that computes gradient of arguments.

    Parameters
    ----------
    func: a perl sub
        The forward (loss) function.
    argnum: an int or arry ref of int
        The index of argument to calculate gradient for.

    Returns
    -------
    grad_func: a perl function
        A function that would compute the gradient of arguments.
=cut


method grad(CodeRef $func, Maybe[Int|ArrayRef[Int]] $argnum=)
{
    my $grad_with_loss_func = __PACKAGE__->grad_and_loss($func, $argnum);
    return sub {
        return ($grad_with_loss_func->(@_))[0];
    };
}

method train_section(CodeRef $sub)
{
    my $prev = __PACKAGE__->set_is_training(1);
    $sub->();
    __PACKAGE__->set_is_training(0) unless $prev;
}

method test_section(CodeRef $sub)
{
    my $prev = __PACKAGE__->set_is_training(0);
    $sub->();
    __PACKAGE__->set_is_training(1) if $prev;
}

1;