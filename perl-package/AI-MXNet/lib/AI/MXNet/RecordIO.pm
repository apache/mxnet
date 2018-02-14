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

package AI::MXNet::RecordIO;
use strict;
use warnings;
use AI::MXNet::Function::Parameters;
use AI::MXNet::Types;
use AI::MXNet::Base;
use Mouse;

=head1 NAME

    AI::MXNet::RecordIO - Read/write RecordIO format data
=cut

=head2 new

    Parameters
    ----------
    uri : Str
        uri path to recordIO file.
    flag: Str
        "r" for reading or "w" writing.
=cut

has 'uri'         => (is => 'ro', isa => 'Str', required => 1);
has 'flag'        => (is => 'ro', isa => enum([qw/r w/]), required => 1);
has 'handle'      => (is => 'rw', isa => 'RecordIOHandle');
has [qw/writable
        is_open/] => (is => 'rw', isa => 'Bool');

sub BUILD
{
    my $self = shift;
    $self->is_open(0);
    $self->open();
}

sub DEMOLISH
{
    shift->close;
}

=head2 open

    Open record file.
=cut

method open()
{
    my $handle;
    if($self->flag eq 'w')
    {
        $handle = check_call(AI::MXNetCAPI::RecordIOWriterCreate($self->uri));
        $self->writable(1);
    }
    else
    {
        $handle = check_call(AI::MXNetCAPI::RecordIOReaderCreate($self->uri));
        $self->writable(0);
    }
    $self->handle($handle);
    $self->is_open(1);
}

=head2 close

    Close record file.
=cut

method close()
{
    return if not $self->is_open;
    if($self->writable)
    {
        check_call(AI::MXNetCAPI::RecordIOWriterFree($self->handle));
    }
    else
    {
        check_call(AI::MXNetCAPI::RecordIOReaderFree($self->handle));
    }
    $self->is_open(0);
}

=head2 reset

    Reset pointer to first item. If record is opened with 'w',
    this will truncate the file to empty.
=cut

method reset()
{
    $self->close;
    $self->open;
}

=head2 write

    Write a string buffer as a record.

    Parameters
    ----------
    $buf : a buffer to write.
=cut

method write(Str $buf)
{
    assert($self->writable);
    check_call(
        AI::MXNetCAPI::RecordIOWriterWriteRecord(
            $self->handle,
            $buf,
            length($buf)
        )
    );
}

=head2 read

    Read a record as a string.

    Returns
    ----------
    $buf : string
=cut

method read()
{
    assert(not $self->writable);
    return scalar(check_call(
        AI::MXNetCAPI::RecordIOReaderReadRecord(
            $self->handle,
        )
    ));
}

method MXRecordIO(@args) { return AI::MXNet::RecordIO->new(uri => $args[0], flag => $args[1]) }
method MXIndexedRecordIO(@args)
{
    return AI::MXNet::IndexedRecordIO->new(
        idx_path => $args[0], uri => $args[1], flag => $args[2]
    )
}

package AI::MXNet::IRHeader;
use Mouse;
has [qw/flag id id2/] => (is => 'rw', isa => 'Int');
has 'label'           => (is => 'rw', isa => 'AcceptableInput');
around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    if(@_ == 4)
    {
        return $class->$orig(flag => $_[0], label => $_[1], id => $_[2], id2 => $_[3]);
    }
    return $class->$orig(@_);
};
my @order = qw/flag label id id2/;
use overload '@{}' => sub { my $self = shift; [map { $self->$_ } @order] };

package AI::MXNet::RecordIO;

=head2 unpack

    unpack a MXImageRecord to a string

    Parameters
    ----------
    s : str
        string buffer from MXRecordIO.read

    Returns
    -------
    header : AI::MXNet::IRHeader
        header of the image record
    s : str
        unpacked string
=cut

method unpack(Str $s)
{
    my $h;
    my $h_size = 24;
    ($h, $s) = (substr($s, 0, $h_size), substr($s, $h_size));
    my $header = AI::MXNet::IRHeader->new(unpack('IfQQ', $h));
    if($header->flag > 0)
    {
        my $label;
        ($label, $s) = (substr($s, 0, 4*$header->flag), substr($s, 4*$header->flag));
        my $pdl_type = PDL::Type->new(DTYPE_MX_TO_PDL->{float32});
        my $pdl = PDL->new_from_specification($pdl_type, $header->flag);
        ${$pdl->get_dataref} = $label;
        $pdl->upd_data;
        $header->label($pdl);
    }
    return ($header, $s)
}

=head2 pack

    pack a string into MXImageRecord

    Parameters
    ----------
    $header : AI::MXNet::IRHeader or ArrayRef suitable for AI::MXNet::IRHeader->new(@{ ArrayRef })
        header of the image record.
        $header->label can be a number or an array ref.
    s : str
        string to pack
=cut

method pack(AI::MXNet::IRHeader|ArrayRef $header, Str $s)
{
    $header = AI::MXNet::IRHeader->new(@$header) unless blessed $header;
    if(not ref $header->label)
    {
        $header->flag(0);
    }
    else
    {
        my $label = AI::MXNet::NDArray->array($header->label, dtype=>'float32')->aspdl;
        $header->label(0);
        $header->flag($label->nelem);
        my $buf = ${$label->get_dataref};
        $s = "$buf$s";
    }
    $s = pack('IfQQ', @{ $header }) . $s;
    return $s;
}

package AI::MXNet::IndexedRecordIO;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::RecordIO';

=head1 NAME

    AI::MXNet::IndexedRecordIO - Read/write RecordIO format data supporting random access.
=cut

=head2 new

    Parameters
    ----------
    idx_path : str
        Path to index file
    uri : str
        Path to record file. Only support file types that are seekable.
    flag : str
        'w' for write or 'r' for read
=cut

has 'idx_path'  => (is => 'ro', isa => 'Str', required => 1);
has [qw/idx
    keys fidx/] => (is => 'rw', init_arg => undef);

method open()
{
    $self->SUPER::open();
    $self->idx({});
    $self->keys([]);
    open(my $f, $self->flag eq 'r' ? '<' : '>', $self->idx_path);
    $self->fidx($f);
    if(not $self->writable)
    {
        while(<$f>)
        {
            chomp;
            my ($key, $val) = split(/\t/);
            push @{ $self->keys }, $key;
            $self->idx->{$key} = $val;
        }
    }
}

method close()
{
    return if not $self->is_open;
    $self->SUPER::close();
    $self->fidx(undef);
}

=head2 seek

    Query current read head position.
=cut

method seek(Int $idx)
{
    assert(not $self->writable);
    my $pos = $self->idx->{$idx};
    check_call(AI::MXNetCAPI::RecordIOReaderSeek($self->handle, $pos));
}

=head2 tell

    Query current write head position.
=cut

method tell()
{
    assert($self->writable);
    return scalar(check_call(AI::MXNetCAPI::RecordIOWriterTell($self->handle)));
}

=head2 read_idx

    Read record with the index.

    Parameters:
    -----------
    $idx
=cut

method read_idx(Int $idx)
{
    $self->seek($idx);
    return $self->read();
}

=head2 write_idx

    Write record with index.

    Parameters:
    -----------
    Int $idx
    Str $buf
=cut

method write_idx(Int $idx, Str $buf)
{
    my $pos = $self->tell();
    $self->write($buf);
    my $f = $self->fidx;
    print $f "$idx\t$pos\n";
    $self->idx->{$idx} = $pos;
    push @{ $self->keys }, $idx;
}

1;
