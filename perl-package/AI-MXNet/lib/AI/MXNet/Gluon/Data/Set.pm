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
package AI::MXNet::Gluon::Data::Set;
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

    AI::MXNet::Gluon::Data::Set
=cut

=head1 DESCRIPTION

    Abstract dataset class. All datasets should have this interface.

    Subclasses need to override method at($i), which returns the i-th
    element, method len() which returns the total number elements.

    AI::MXNet::NDArray can be directly used as a dataset.
=cut

method at(Index $idx) { confess("Not Implemented") }

method len() { confess("Not Implemented") }

package AI::MXNet::Gluon::Data::ArrayDataset;
use AI::MXNet::Base;
use Mouse;
extends 'AI::MXNet::Gluon::Data::Set';

=head1 NAME

    AI::MXNet::Gluon::Data::ArrayDataset
=cut

=head1 DESCRIPTION

    A dataset with a data array and a label array.

    The i-th sample is `(data[i], label[i])`.

    Parameters
    ----------
    data : AI::MXNet::NDArray or PDL
        The data array.
    label : AI::MXNet::NDArray or PDL
        The label array.
=cut
has [qw/data label/] => (is => 'rw', isa => 'PDL|AI::MXNet::NDArray', required => 1);
method python_constructor_arguments() { ['data', 'label'] }

sub BUILD
{
    my $self = shift;
    assert(($self->data->len == $self->label->len), "data and label lengths must be the same");
    if($self->label->isa('AI::MXNet::NDArray') and @{$self->label->shape} == 1)
    {
        $self->label($self->label->aspdl);
    }
    if($self->data->isa('PDL'))
    {
        $self->data(AI::MXNet::NDArray->array($self->data));
    }
}

method at(Index $idx)
{
    return [
        $self->data->at($idx),
        $self->label->at($idx)
    ];
}

method len()
{
    return $self->data->len
}

__PACKAGE__->register('AI::MXNet::Gluon::Data');

package AI::MXNet::Gluon::Data::RecordFileSet;
use AI::MXNet::Base;
use Mouse;
extends 'AI::MXNet::Gluon::Data::Set';

=head1 NAME

    AI::MXNet::Gluon::Data::RecordFileSet
=cut

=head1 DESCRIPTION

    A dataset wrapping over a RecordIO (.rec) file.

    Each sample is a string representing the raw content of an record.

    Parameters
    ----------
    filename : str
        Path to rec file.
=cut
has 'filename' => (is => 'ro', isa =>'Str', required => 1);
has '_record'  => (is => 'rw', init_arg => undef);
method python_constructor_arguments() { ['filename'] }

sub BUILD
{
    my $self = shift;
    my $idx_file = $self->filename;
    $idx_file =~ s/\.[^.]+$/.idx/;
    $self->_record(
        AI::MXNet::IndexedRecordIO->new(
            idx_path => $idx_file, uri => $self->filename, flag => 'r'
        )
    );
}

method at(Index $idx) { return $self->_record->read_idx($idx); }

method len() { return scalar(@{ $self->_record->keys }) }

__PACKAGE__->register('AI::MXNet::Gluon::Data');

1;