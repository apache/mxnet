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

=head1 NAME

    AI::MXNet::Gluon::Data::Loader::DataLoader - Dataset generator.
=cut

=head1 DESCRIPTION

    Loads data from a dataset and returns mini-batches of data.

    Parameters
    ----------
    dataset : Dataset
        Source dataset. Note that numpy and mxnet arrays can be directly used
        as a Dataset.
    batch_size : int
        Size of mini-batch.
    shuffle : bool
        Whether to shuffle the samples.
    sampler : Sampler
        The sampler to use. Either specify sampler or shuffle, not both.
    last_batch : {'keep', 'discard', 'rollover'}
        How to handle the last batch if batch_size does not evenly divide
        `len(dataset)`.

        keep - A batch with less samples than previous batches is returned.
        discard - The last batch is discarded if its incomplete.
        rollover - The remaining samples are rolled over to the next epoch.
    batch_sampler : Sampler
        A sampler that returns mini-batches. Do not specify batch_size,
        shuffle, sampler, and last_batch if batch_sampler is specified.
=cut

use strict;
use warnings;
package AI::MXNet::Gluon::Data::Loader::DataLoader;
use AI::MXNet::Function::Parameters;
use Mouse;

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

# Collate data into batch.
func _batchify($data, $dtype)
{
    if(blessed $data->[0] and $data->[0]->isa('AI::MXNet::NDArray'))
    {
        return AI::MXNet::NDArray->stack(@{ $data });
    }
    elsif(ref $data->[0] eq 'ARRAY')
    {
        my (@data, @label);
        for my $i (@$data)
        {
            my ($d, $l) = @$i;
            push @data, $d;
            push @label, $l;
        }
        return [_batchify(\@data, $dtype), _batchify(\@label, $dtype)];
    }
    else
    {
        return AI::MXNet::NDArray->array($data, dtype => $dtype);
    }
}

has 'dataset'       => (is => 'rw', isa => 'AI::MXNet::Gluon::Data::Set|AI::MXNet::NDArray|PDL', required => 1);
has 'batch_size'    => (is => 'ro', isa => 'Int');
has 'shuffle'       => (is => 'ro', isa => 'Bool', default => 0);
has 'sampler'       => (is => 'rw', isa => 'AI::MXNet::Gluon::Data::Sampler');
has 'batch_sampler' => (is => 'rw', isa => 'AI::MXNet::Gluon::Data::Sampler');
has 'last_batch'    => (is => 'rw', isa => 'Str', default => 'keep');

around BUILDARGS => \&AI::MXNet::Base::process_arguments;
method python_constructor_arguments() { ['dataset', 'batch_size'] }

sub BUILD
{
    my $self = shift;
    if($self->dataset->isa('PDL'))
    {
        $self->dataset(AI::MXNet::NDArray->array($self->dataset));
    }
    if(not defined $self->batch_sampler)
    {
        if(not defined $self->batch_size)
        {
            confess("batch_size must be specified unless batch_sampler is specified");
        }
        if(not defined $self->sampler)
        {
            if($self->shuffle)
            {
                $self->sampler(
                    AI::MXNet::Gluon::Data::Sampler::RandomSampler->new(
                        length => $self->dataset->len
                    )
                );
            }
            else
            {
                $self->sampler(
                    AI::MXNet::Gluon::Data::Sampler::SequentialSampler->new(
                        length => $self->dataset->len,
                    )
                );
            }
        }
        elsif($self->shuffle)
        {
            confess("shuffle must not be specified if sampler is specified");
        }
        $self->batch_sampler(
            AI::MXNet::Gluon::Data::Sampler::BatchSampler->new(
                sampler => $self->sampler,
                batch_size => $self->batch_size,
                last_batch => $self->last_batch
            )
        );
    }
    elsif(defined $self->batch_size or $self->shuffle or defined $self->sampler or defined $self->last_batch)
    {
        confess("batch_size, shuffle, sampler and last_batch must ".
                "not be specified if batch_sampler is specified.");
    }
}

use overload
    '<>' => sub {
        my $self = shift;
        my $sampler = $self->batch_sampler;
        my $batch = <$sampler>;
        if(not defined $batch)
        {
            return undef;
        };
        return _batchify([map { $self->dataset->at($_) } @{ $batch }], eval { $self->dataset->label->dtype }//'int32');
    };

method len()
{
    $self->batch_sampler->len;
}

use overload '@{}' => sub { shift->list };

method list()
{
    my @ret;
    while(defined(my $data = <$self>))
    {
        push @ret, $data;
    }
    return \@ret;
}

__PACKAGE__->register('AI::MXNet::Gluon::Data');

1;