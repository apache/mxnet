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

     Compute the gradients of outputs w.r.t variables.

     Parameters
     ----------
     outputs: array ref of NDArray
     out_grads: array ref of NDArray or undef
     retain_graph: bool, defaults to false
     train_mode: bool, defaluts to true
=cut

method backward(
    ArrayRef[AI::MXNet::NDArray] $outputs,
    Maybe[ArrayRef[AI::MXNet::NDArray|Undef]] $out_grads=,
    Bool $retain_graph=0,
    Bool $train_mode=1
)
{
    my @output_handles = map { $_->handle } @{ $outputs };
    if(not defined $out_grads)
    {
        check_call(
            AI::MXNetCAPI::AutogradBackwardEx(
                scalar(@output_handles),
                \@output_handles,
                [],
                $retain_graph,
                $train_mode
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
        AI::MXNetCAPI::AutogradBackwardEx(
            scalar(@output_handles),
            \@output_handles,
            \@ograd_handles,
            $retain_graph,
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

1;