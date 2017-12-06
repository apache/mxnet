use strict;
use warnings;
use Test::More;
use AI::MXNet qw(mx);

sub test_builtin_zip()
{
    is_deeply(
        [ AI::MXNet::zip([ 0 .. 9 ], [ 10 .. 19 ]) ],
        [ map { [ $_, 10 + $_ ] } 0 .. 9 ]);
    is_deeply(
        [ AI::MXNet::zip([ 0 .. 9 ], [ 10 .. 19 ], [ 20 .. 29 ]) ],
        [ map { [ $_, 10 + $_, 20 + $_ ] } 0 .. 9 ]);
    my $over = ListOverload->new(10 .. 19);
    is_deeply(
        [ AI::MXNet::zip([ 0 .. 9 ], \@$over) ],
        [ map { [ $_, 10 + $_ ] } 0 .. 9 ]);
    my $tied = ListTied->new(10 .. 19);
    is_deeply(
        [ AI::MXNet::zip([ 0 .. 9 ], \@$tied) ],
        [ map { [ $_, 10 + $_ ] } 0 .. 9 ]);
}


test_builtin_zip();
done_testing();

package ListTied {
    sub new {
        my($class, @list) = @_;
        my @tied;
        tie @tied, $class, @list;
        return \@tied;
    }
    sub TIEARRAY {
        my($class, @list) = @_;
        return bless { list => \@list }, $class;
    }
    sub FETCH {
        my($self, $index) = @_;
        return $self->{list}[$index];
    }
    sub STORE {
        my($self, $index, $value) = @_;
        return $self->{list}[$index] = $value;
    }
    sub FETCHSIZE {
        my($self) = @_;
        return scalar @{$self->{list}};
    }
    sub STORESIZE {
        my($self, $count) = @_;
        return $self->{list}[$count - 1] //= undef;
    }
    sub EXTEND {
        my($self, $count) = @_;
        return $self->STORESIZE($count);
    }
    sub EXISTS {
        my($self, $key) = @_;
        return exists $self->{list}[$key];
    }
    sub DELETE {
        my($self, $key) = @_;
        return delete $self->{list}[$key];
    }
    sub CLEAR {
        my($self) = @_;
        return @{$self->{list}} = ();
    }
    sub PUSH {
        my($self, @list) = @_;
        return push @{$self->{list}}, @list;
    }
    sub POP {
        my($self) = @_;
        return pop @{$self->{list}};
    }
    sub SHIFT {
        my($self) = @_;
        return shift @{$self->{list}};
    }
    sub UNSHIFT {
        my($self, @list) = @_;
        return unshift @{$self->{list}}, @list;
    }
    sub SPLICE {
        my($self, $offset, $length, @list) = @_;
        return splice @{$self->{list}}, $offset, $length, @list;
    }
    sub UNTIE {
        my($self) = @_;
    }
    sub DESTROY {
        my($self) = @_;
    }
}

package ListOverload {
    use overload '@{}' => \&as_list;
    sub new {
        my($class, @list) = @_;
        return bless { list => \@list }, $class;
    }
    sub as_list { return $_[0]{list} }
}

