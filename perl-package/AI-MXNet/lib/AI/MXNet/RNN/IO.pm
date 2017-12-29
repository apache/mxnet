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

package AI::MXNet::RNN::IO;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;

=encoding UTF-8

=head1 NAME

    AI::MXNet::RNN::IO - Functions for constructing recurrent neural networks.
=cut

=head1 DESCRIPTION

    Functions for constructing recurrent neural networks.
=cut

=head2 encode_sentences

    Encode sentences and (optionally) build a mapping
    from string tokens to integer indices. Unknown keys
    will be added to vocabulary.

    Parameters
    ----------
    $sentences : array ref of array refs of str
        A array ref of sentences to encode. Each sentence
        should be a array ref of string tokens.
    :$vocab : undef or hash ref of str -> int
        Optional input Vocabulary
    :$invalid_label : int, default -1
        Index for invalid token, like <end-of-sentence>
    :$invalid_key : str, default '\n'
        Key for invalid token. Uses '\n' for end
        of sentence by default.
    :$start_label=0 : int
        lowest index.

    Returns
    -------
    $result : array ref of array refs of int
        encoded sentences
    $vocab : hash ref of str -> int
        result vocabulary
=cut


method encode_sentences(
    ArrayRef[ArrayRef]  $sentences,
    Maybe[HashRef]     :$vocab=,
    Int                :$invalid_label=-1,
    Str                :$invalid_key="\n",
    Int                :$start_label=0
)
{
    my $idx = $start_label;
    my $new_vocab;
    if(not defined $vocab)
    {
        $vocab = { $invalid_key => $invalid_label };
        $new_vocab = 1;
    }
    else
    {
        $new_vocab = 0;
    }
    my @res;
    for my $sent (@{ $sentences })
    {
        my @coded;
        for my $word (@{ $sent })
        {
            if(not exists $vocab->{ $word })
            {
                assert($new_vocab, "Unknown token: $word");
                if($idx == $invalid_label)
                {
                    $idx += 1;
                }
                $vocab->{$word} = $idx;
                $idx += 1;
            }
            push @coded, $vocab->{ $word };
        }
        push @res, \@coded;
    }
    return (\@res, $vocab);
}

package AI::MXNet::BucketSentenceIter;

=encoding UTF-8

=head1 NAME

    AI::MXNet::BucketSentenceIter
=cut

=head1 DESCRIPTION

    Simple bucketing iterator for language model.
    Label for each step is constructed from data of
    next step.
=cut

=head2 new

    Parameters
    ----------
    sentences : array ref of array refs of int
        encoded sentences
    batch_size : int
        batch_size of data
    invalid_label : int, default -1
        key for invalid label, e.g. <end-of-sentence>
    dtype : str, default 'float32'
        data type
    buckets : array ref of int
        size of data buckets. Automatically generated if undef.
    data_name : str, default 'data'
        name of data
    label_name : str, default 'softmax_label'
        name of label
    layout : str
        format of data and label. 'NT' means (batch_size, length)
        and 'TN' means (length, batch_size).
=cut

use Mouse;
use AI::MXNet::Base;
use List::Util qw(shuffle max);
extends 'AI::MXNet::DataIter';
has 'sentences'     => (is => 'ro', isa => 'ArrayRef[ArrayRef]', required => 1);
has '+batch_size'   => (is => 'ro', isa => 'Int',                required => 1);
has 'invalid_label' => (is => 'ro', isa => 'Int',   default => -1);
has 'data_name'     => (is => 'ro', isa => 'Str',   default => 'data');
has 'label_name'    => (is => 'ro', isa => 'Str',   default => 'softmax_label');
has 'dtype'         => (is => 'ro', isa => 'Dtype', default => 'float32');
has 'layout'        => (is => 'ro', isa => 'Str',   default => 'NT');
has 'buckets'       => (is => 'rw', isa => 'Maybe[ArrayRef[Int]]');
has [qw/data nddata ndlabel
        major_axis default_bucket_key
        provide_data provide_label
        idx curr_idx
    /]              => (is => 'rw', init_arg => undef);

sub BUILD
{
    my $self = shift;
    if(not defined $self->buckets)
    {
        my @buckets;
        my $p = pdl([map { scalar(@$_) } @{ $self->sentences }]);
        enumerate(sub {
            my ($i, $j) = @_;
            if($j >= $self->batch_size)
            {
                push @buckets, $i;
            }
        }, $p->histogram(1,0,$p->max+1)->unpdl);
        $self->buckets(\@buckets);
    }
    @{ $self->buckets } = sort { $a <=> $b } @{ $self->buckets };
    my $ndiscard = 0;
    $self->data([map { [] } 0..@{ $self->buckets }-1]);
    for my $i (0..@{$self->sentences}-1)
    {
        my $buck = bisect_left($self->buckets, scalar(@{ $self->sentences->[$i] }));
        if($buck == @{ $self->buckets })
        {
            $ndiscard += 1;
            next;
        }
        my $buff = AI::MXNet::NDArray->full(
            [$self->buckets->[$buck]],
            $self->invalid_label,
            dtype => $self->dtype
        )->aspdl;
        $buff->slice([0, @{ $self->sentences->[$i] }-1]) .= pdl($self->sentences->[$i]);
        push @{ $self->data->[$buck] }, $buff;
    }
    $self->data([map { pdl(PDL::Type->new(DTYPE_MX_TO_PDL->{$self->dtype}), $_) } @{$self->data}]);
    AI::MXNet::Logging->warning("discarded $ndiscard sentences longer than the largest bucket.")
        if $ndiscard;
    $self->nddata([]);
    $self->ndlabel([]);
    $self->major_axis(index($self->layout, 'N'));
    $self->default_bucket_key(max(@{ $self->buckets }));
    my $shape;
    if($self->major_axis == 0)
    {
        $shape = [$self->batch_size, $self->default_bucket_key];
    }
    elsif($self->major_axis == 1)
    {
        $shape = [$self->default_bucket_key, $self->batch_size];
    }
    else
    {
        confess("Invalid layout ${\ $self->layout }: Must by NT (batch major) or TN (time major)");
    }
    $self->provide_data([
        AI::MXNet::DataDesc->new(
            name  => $self->data_name,
            shape => $shape,
            dtype => $self->dtype,
            layout => $self->layout
        )
    ]);
    $self->provide_label([
        AI::MXNet::DataDesc->new(
            name  => $self->label_name,
            shape => $shape,
            dtype => $self->dtype,
            layout => $self->layout
        )
    ]);
    $self->idx([]);
    enumerate(sub {
        my ($i, $buck) = @_;
        my $buck_len = $buck->shape->at(-1);
        for my $j (0..($buck_len - $self->batch_size))
        {
            if(not $j%$self->batch_size)
            {
                push @{ $self->idx }, [$i, $j];
            }
        }
    }, $self->data);
    $self->curr_idx(0);
    $self->reset;
}

method reset()
{
    $self->curr_idx(0);
    @{ $self->idx } = shuffle(@{ $self->idx });
    $self->nddata([]);
    $self->ndlabel([]);
    for my $buck (@{ $self->data })
    {
        $buck = pdl_shuffle($buck);
        my $label = $buck->zeros;
        $label->slice([0, -2], 'X')  .= $buck->slice([1, -1], 'X');
        $label->slice([-1, -1], 'X') .= $self->invalid_label;
        push @{ $self->nddata }, AI::MXNet::NDArray->array($buck, dtype => $self->dtype);
        push @{ $self->ndlabel }, AI::MXNet::NDArray->array($label, dtype => $self->dtype);
    }
}

method next()
{
    return undef if($self->curr_idx == @{ $self->idx });
    my ($i, $j) = @{ $self->idx->[$self->curr_idx] };
    $self->curr_idx($self->curr_idx + 1);
    my ($data, $label);
    if($self->major_axis == 1)
    {
        $data  = $self->nddata->[$i]->slice([$j, $j+$self->batch_size-1])->T;
        $label = $self->ndlabel->[$i]->slice([$j, $j+$self->batch_size-1])->T;
    }
    else
    {
        $data = $self->nddata->[$i]->slice([$j, $j+$self->batch_size-1]);
        $label = $self->ndlabel->[$i]->slice([$j, $j+$self->batch_size-1]);
    }
    return AI::MXNet::DataBatch->new(
        data          => [$data],
        label         => [$label],
        bucket_key    => $self->buckets->[$i],
        pad           => 0,
        provide_data  => [
            AI::MXNet::DataDesc->new(
                name  => $self->data_name,
                shape => $data->shape,
                dtype => $self->dtype,
                layout => $self->layout
            )
        ],
        provide_label => [
            AI::MXNet::DataDesc->new(
                name  => $self->label_name,
                shape => $label->shape,
                dtype => $self->dtype,
                layout => $self->layout
            )
        ],
    );
}

1;
