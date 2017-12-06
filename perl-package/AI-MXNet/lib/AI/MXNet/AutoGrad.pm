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

package AI::MXNet::AutoGrad;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;
use Scalar::Util qw(blessed);
use Carp qw(confess);

sub import
{
    my ($class, $short_name) = @_;
    if($short_name)
    {
        $short_name =~ s/[^\w:]//g;
        if(length $short_name)
        {
            my $short_name_package =<<"EOP";
            package $short_name;
            use parent 'AI::MXNet::AutoGrad';
            1;
EOP
            eval $short_name_package;
        }
    }
}

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
    return scalar(check_call(AI::MXNetCAPI::AutogradSetIsTraining($is_train)));
}

=head2 set_is_recording

    Set status to recording/not recording. When recording, graph will be constructed
    for gradient computation.

    Parameters
    ----------
    is_recoding: bool

    Returns
    -------
    previous state before this set.
=cut

method set_is_recording(Bool $is_recording)
{
    return scalar(check_call(AI::MXNetCAPI::AutogradSetIsRecording($is_recording)));
}

=head2 is_recording

    Get status on recording/not recording.

    Returns
    -------
    Current state of recording.
=cut

method is_recording()
{
    return scalar(check_call(AI::MXNetCAPI::AutogradIsRecording()));
}

=head2 is_training

    Get status on training/predicting.

    Returns
    -------
    Current state of training/predicting.
=cut

method is_training()
{
    return scalar(check_call(AI::MXNetCAPI::AutogradIsTraining()));
}

=head2 mark_variables

    Mark AI::MXNet::NDArrays as variables to compute gradient for autograd.

    Parameters
    ----------
    ArrayRef[AI::MXNet::NDArray] $variables
    ArrayRef[AI::MXNet::NDArray] $gradients
    GradReq|ArrayRef[GradReq]   :$grad_reqs='write'
=cut

method mark_variables(
    ArrayRef[AI::MXNet::NDArray]  $variables,
    ArrayRef[AI::MXNet::NDArray]  $gradients,
    GradReq|ArrayRef[GradReq]    :$grad_reqs='write'
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

    Compute the gradients of heads w.r.t previously marked variables.

    Parameters
    ----------
    $heads: ArrayRef[AI::MXNet::NDArray]
        Output NDArray(s)
    :$head_grads=: Maybe[AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray|Undef]]
        Gradients with respect to heads.
    :$retain_graph=0: bool, optional
        Whether to retain graph.
    :$train_mode=1: bool, optional
        Whether to do backward for training or predicting.
=cut
method backward(
    AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray] $heads,
    Maybe[AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray|Undef]] :$head_grads=,
    Bool :$retain_graph=0,
    Bool :$train_mode=1
)
{
    my ($head_handles, $hgrad_handles) = _parse_head($heads, $head_grads);
    check_call(
        AI::MXNetCAPI::AutogradBackwardEx(
            scalar(@{ $head_handles }),
            $head_handles,
            $hgrad_handles,
            0,
            [],
            $retain_graph,
            0,
            $train_mode
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
            @variables = map { $args[$_] } @argnum;
        }
        map {
            assert(
                (blessed($_) and $_->isa('AI::MXNet::NDArray')),
                "type of autograd input should NDArray")
        } @variables;
        my @grads = map { $_->zeros_like } @variables;
        __PACKAGE__->mark_variables(\@variables, \@grads);
        my $outputs;
        __PACKAGE__->record(sub { $outputs = $func->(@args) });
        __PACKAGE__->backward(ref $outputs eq 'ARRAY' ? $outputs : [$outputs]);
        return (\@grads, $outputs);
    };
}

=head2 grad

    Compute the gradients of heads w.r.t variables. Gradients will be
    returned as new NDArrays instead of stored into `variable.grad`.
    Supports recording gradient graph for computing higher order gradients.

    .. Note: Currently only a very limited set of operators support higher order
    gradients.

    Parameters
    ----------
    $heads: NDArray or array ref of NDArray
        Output NDArray(s)
    $variables: NDArray or list of NDArray
        Input variables to compute gradients for.
    :$head_grads=: NDArray or list of NDArray or undef
        Gradients with respect to heads.
    :$retain_graph=: bool
        Whether to keep computation graph to differentiate again, instead
        of clearing history and release memory. Defaults to the same value
        as create_graph.
    :$create_graph=0: bool
        Whether to record gradient graph for computing higher order
    $train_mode=1: bool, optional
        Whether to do backward for training or prediction.

    Returns
    -------
    NDArray or list of NDArray:
        Gradients with respect to variables.

    Examples
    --------
    >>> $x = mx->nd->ones([1]);
    >>> $x->attach_grad();
    >>> mx->autograd->record(sub {
            $z = mx->nd->elemwise_add(mx->nd->exp($x), $x);
        });
    >>> $dx = mx->autograd->grad($z, [$x], create_graph=>1)
    >>> $dx->backward();
    >>> print($dx->grad->aspdl)
    [3.71828175]
=cut

method grad(
    AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray] $heads,
    AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray] $variables,
    Maybe[AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray|Undef]] :$head_grads=,
    Bool :$retain_graph=,
    Bool :$create_graph=0,
    Bool :$train_mode=1
)
{
    my ($head_handles, $hgrad_handles) = _parse_head($heads, $head_grads);
    my @var_handles;
    if(blessed $variables)
    {
        @var_handles = ($variables->handle);
    }
    else
    {
        assert(scalar(@{ $variables }), "variables cannot be an empty array.");
        @var_handles = map { $_->handle } @{ $variables };
    }

    $retain_graph //= $create_graph;

    my ($grad_vars, $grad_stypes)
        =
    check_call(
        AI::MXNetCAPI::AutogradBackwardEx(
            scalar(@{ $head_handles }),
            $head_handles,
            $hgrad_handles,
            scalar(@var_handles),
            \@var_handles,
            $retain_graph,
            $create_graph,
            $train_mode
        )
    );

    my @ret;
    for(zip($grad_vars, $grad_stypes)) {
        my ($handle, $stype) = @$_;
        push @ret, AI::MXNet::NDArray->new(handle => $handle, stype => $stype);
    }
    if(blessed $variables)
    {
        return $ret[0];
    }
    return \@ret;
}

=head2 train_mode

    Executes $sub within an autograd training scope context.
    Parameters
    ----------
    CodeRef $sub: a perl sub
=cut

method train_mode(CodeRef $sub)
{
    my $prev = __PACKAGE__->set_is_training(1);
    eval { $sub->(); };
    __PACKAGE__->set_is_training(0) unless $prev;
    confess($@) if $@;
}

=head2 predict_mode

    Executes $sub within an autograd predicting scope context.
    Parameters
    ----------
    CodeRef $sub: a perl sub
=cut

method predict_mode(CodeRef $sub)
{
    my $prev = __PACKAGE__->set_is_training(0);
    eval { $sub->(); };
    __PACKAGE__->set_is_training(1) if $prev;
    confess($@) if $@;
}

=head2 record

    Executes $sub within an autograd recording scope context
    and captures code that needs gradients to be calculated.
    Parameters
    ----------
    CodeRef $sub: a perl sub
    Maybe[Bool] :$train_mode=1
=cut

method record(CodeRef $sub, Maybe[Bool] :$train_mode=1)
{
    my $prev_train;
    if(defined $train_mode)
    {
        $prev_train = __PACKAGE__->set_is_training($train_mode);
    }
    my $prev_recording = __PACKAGE__->set_is_recording(1);
    eval { $sub->(); };
    if(defined $train_mode)
    {
        $prev_train = __PACKAGE__->set_is_training($prev_train) if not $prev_train == $train_mode;
    }
    __PACKAGE__->set_is_recording(0) unless $prev_recording;
    confess($@) if $@;
}

=head2 pause

    Executes $sub within an autograd recording scope context
    and captures code that needs gradients to be calculated.
    Parameters
    ----------
    CodeRef $sub: a perl sub
    Maybe[Bool] :$train_mode=0
=cut

method pause(CodeRef $sub, Maybe[Bool] :$train_mode=0)
{
    my $prev_train;
    if(defined $train_mode)
    {
        $prev_train = __PACKAGE__->set_is_training($train_mode);
    }
    my $prev_recording = __PACKAGE__->set_is_recording(0);
    eval { $sub->(); };
    if(defined $train_mode)
    {
        $prev_train = __PACKAGE__->set_is_training($prev_train) if not $prev_train == $train_mode;
    }
    __PACKAGE__->set_is_recording(1) if $prev_recording;
    confess($@) if $@;
}

=head2 get_symbol

    Retrieve recorded computation history as `Symbol`.

    Parameters
    ----------
    x : NDArray
        Array representing the head of computation graph.
    Returns
    -------
    Symbol
        The retrieved Symbol.
=cut

method get_symbol(AI::MXNet::NDArray $x)
{
    my $handle = scalar(check_call(AI::MXNetCAPI::AutogradGetSymbol($x->handle)));
    return AI::MXNet::Symbol->new(handle => $handle);
}

# parse head gradient for backward and grad.
func _parse_head($heads, $head_grads)
{
    if(blessed $heads)
    {
        $heads = [$heads];
    }
    if(blessed $head_grads)
    {
        $head_grads = [$head_grads];
    }
    my @head_handles = map { $_->handle } @{ $heads };
    my @hgrad_handles;
    if(defined $head_grads)
    {
        assert(
            (@{ $heads } == @{ $head_grads }),
            "heads and head_grads must be lists of the same length"
        );
        @hgrad_handles = map { defined($_) ? $_->handle : undef } @{ $head_grads };
    }
    return (\@head_handles, \@hgrad_handles);
}

1;
