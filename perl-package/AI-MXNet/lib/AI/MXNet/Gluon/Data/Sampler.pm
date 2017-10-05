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

use strict;
use warnings;

package AI::MXNet::Gluon::Data::Sampler;
use AI::MXNet::Function::Parameters;
use Mouse;
around BUILDARGS => \&AI::MXNet::Base::process_arguments;

method _class_name()
{
    my $class = ref $self || $self;
    $class =~ s/^.+:://;
    $class;
}

method register(Str $container)
{
    my $sub_name = $self->_class_name;
    no strict 'refs';
    *{$container.'::'.$sub_name} = sub { shift; $self->new(@_) };
}

=head1 NAME

    AI::MXNet::Gluon::Data::Sampler
=cut

=head1 DESCRIPTION

    Base class for samplers.

    All samplers should subclass AI::MXNet::Gluon::Data::Sampler 
    and define method 'len' and 'next'
    methods.
=cut

use overload '<>' =>  sub { shift->next },
             '@{}' => sub { shift->list };

method list()
{
    my @ret;
    while(defined(my $data = <$self>))
    {
        push @ret, $data;
    }
    return \@ret;
}

method len() { confess('Not Implemented') }
method next() { confess('Not Implemented') }

package AI::MXNet::Gluon::Data::Sampler::SequentialSampler;
use Mouse;
extends 'AI::MXNet::Gluon::Data::Sampler';

=head1 NAME

    AI::MXNet::Gluon::Data::Sampler::SequentialSampler
=cut

=head1 DESCRIPTION

    Samples elements from [0, length) sequentially.

    Parameters
    ----------
    length : int
        Length of the sequence.
=cut
has 'length'   => (is => 'ro', isa => 'Int', required => 1);
has '_current' => (is => 'rw', init_arg => undef, default => 0);
method python_constructor_arguments() { ['length'] }

method next()
{
    my $current = $self->_current;
    if($self->_current == $self->length)
    {
        $self->reset;
        return undef;
    }
    else
    {
        $self->_current($self->_current + 1);
        return $current;
    }
};

method reset() { $self->_current(0) }
method len() { $self->length }

__PACKAGE__->register('AI::MXNet::Gluon::Data');

package AI::MXNet::Gluon::Data::Sampler::RandomSampler;
use Mouse;
use List::Util qw(shuffle);
extends 'AI::MXNet::Gluon::Data::Sampler';

=head1 NAME

    AI::MXNet::Gluon::Data::Sampler::RandomSampler
=cut

=head1 DESCRIPTION

    Samples elements from [0, length) randomly without replacement.

    Parameters
    ----------
    length : int
        Length of the sequence.
=cut
has 'length'   => (is => 'ro', isa => 'Int', required => 1);
has '_current' => (is => 'rw', init_arg => undef, default => 0);
has '_indices' => (is => 'rw', init_arg => undef);
method python_constructor_arguments() { ['length'] }

sub BUILD
{
    my $self = shift;
    $self->_indices([shuffle(0..$self->length-1)]);
}

method next()
{
    my $current = $self->_current;
    if($self->_current == $self->length)
    {
        $self->reset;
        return undef;
    }
    else
    {
        $self->_current($self->_current + 1);
        return $self->_indices->[$current];
    }
};

method reset() { @{ $self->_indices } = shuffle(@{ $self->_indices }); $self->_current(0) }
method len() { $self->length }

__PACKAGE__->register('AI::MXNet::Gluon::Data');

package AI::MXNet::Gluon::Data::Sampler::BatchSampler;
use Mouse;
use List::Util qw(shuffle);
extends 'AI::MXNet::Gluon::Data::Sampler';

=head1 NAME

    AI::MXNet::Gluon::Data::Sampler::BatchSampler
=cut

=head1 DESCRIPTION

    Wraps over another AI::MXNet::Gluon::Data::Sampler and return mini-batches of samples.

    Parameters
    ----------
    sampler : AI::MXNet::Gluon::Data::Sampler
        The source Sampler.
    batch_size : int
        Size of mini-batch.
    last_batch : {'keep', 'discard', 'rollover'}
        Specifies how the last batch is handled if batch_size does not evenly
        divide sequence length.

        If 'keep', the last batch will be returned directly, but will contain
        less element than `batch_size` requires.

        If 'discard', the last batch will be discarded.

        If 'rollover', the remaining elements will be rolled over to the next
        iteration.

    Examples
    --------
    >>> $sampler = gluon->data->SequentialSampler(10)
    >>> $batch_sampler = gluon->data->BatchSampler($sampler, batch_size => 3, last_batch => 'keep');
    >>> @{ $batch_sampler }
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
=cut
has 'batch_size' => (is => 'ro', isa => 'Int', required => 1);
has 'sampler'    => (is => 'ro', isa => 'AI::MXNet::Gluon::Data::Sampler', required => 1);
has 'last_batch' => (is => 'ro', isa => 'Str', default => 'keep');
has '_prev'      => (is => 'rw', init_arg => undef);
has '_kept'      => (is => 'rw', init_arg => undef);
method python_constructor_arguments() { ['sampler', 'batch_size', 'last_batch'] }

sub BUILD
{
    my $self = shift;
    $self->_prev([]);
}

method next()
{
    if($self->_kept)
    {
        $self->_kept(0);
        return undef;
    }
    $self->_kept(0);
    my $batch = $self->_prev;
    $self->_prev([]);
    my $sampler = $self->sampler;
    while(defined(my $i = <$sampler>))
    {
        push @{ $batch }, $i;
        if(@{ $batch } == $self->batch_size)
        {
            return $batch;
        }
    }
    if(@{ $batch })
    {
        if($self->last_batch eq 'keep')
        {
            $self->_kept(1);
            return $batch;
        }
        elsif($self->last_batch eq 'discard')
        {
            return undef;
        }
        elsif($self->last_batch eq 'rollover')
        {
            $self->_prev($batch);
            return undef;
        }
        else
        {
            confess(
                "last_batch must be one of 'keep', 'discard', or 'rollover', ".
                "but got ${\ $self->last_batch }"
            );
        }
    }
    return undef;
}

method len()
{
    if($self->last_batch eq 'keep')
    {
        return int(($self->sampler->len + $self->batch_size - 1) / $self->batch_size);
    }
    elsif($self->last_batch eq 'discard')
    {
        return int($self->sampler->len/$self->batch_size);
    }
    elsif($self->last_batch eq 'rollover')
    {
        return int((@{ $self->_prev } + $self->sampler->len) / $self->batch_size);
    }
    else
    {
        confess(
            "last_batch must be one of 'keep', 'discard', or 'rollover', ".
            "but got ${\ $self->last_batch }"
        );
    }
}

__PACKAGE__->register('AI::MXNet::Gluon::Data');

1;